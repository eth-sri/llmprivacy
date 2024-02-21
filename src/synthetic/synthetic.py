import json
import os
import re
from src.configs import Config
from src.models.model_factory import get_model
from src.prompts import Prompt, Conversation
from .synthetic_bots_prompt_builder import build_user_bot_prompt, build_investigator_bot_prompt
from src.chat.chat_parser import extract_predictions
from .eval_tools import evaluate_guess_synthetic


def load_txt(path: str) -> str:
    """
    Takes a path to a .txt and loads it into a string.

    :param path: The path to the .txt file.
    :return: The .txt loaded into a string.
    """
    with open(path, 'r') as f:
        string = f.read()
    
    return string


def get_trial(sample_path: str) -> str:
    """
    Takes the path of the sample and return the fitting sample number.

    :param sample_path: The path to the sample, which will be split to extract the parent directory.
    :return: The number of the sample.
    """
    # split the sample path to get the parent directory
    parent_dir = '/'.join([s for s in sample_path.split('/') if not s.endswith('.json')])

    # list all files in the parent directory
    files = os.listdir(parent_dir)
    prefix = '_'.join([s for s in sample_path.split('/')[-1].split('_') if not s.endswith('.json')])
    pattern = re.compile(rf'{prefix}_([0-9]+)\.json$')

    nums = []
    for file in files:
        
        match = pattern.search(file)
        if match:
            num = int(match.group(1))
            nums.append(num)

    if nums:
        return max(nums) + 1
    else:
        return 0


def build_paths(cfg: Config, hardness: int) -> dict[str, str]:
    """
    Takes the config of the task, and returns a dictionary containing the paths of the user-bot, 
    investigator-bot, and the evaluator-bot system prompts, the paths to the examples, and the 
    path under which the samples will be saved.

    :param cfg: The task-config containing all necessary information for building the paths.
    :param hardness: The hardness of the examples to load.
    :return: A dictionary of the paths.    
    """
    return_paths = {}

    # bot prompts
    return_paths['user_bot_base_prompt'] = cfg.task_config.base_path + '/base_prompts/user_bot_base_prompt.txt'
    return_paths['investigator_bot_base_prompt'] = cfg.task_config.base_path + '/base_prompts/investigator_bot_base_prompt.txt'

    user_bot_name, investigator_bot_name = cfg.task_config.user_bot.get_name(), cfg.task_config.investigator_bot.get_name()
    
    # save paths and example paths
    for feature in cfg.task_config.features:

        # samples
        base_feature_path = cfg.task_config.base_path + f'/samples/{investigator_bot_name}-{user_bot_name}/{feature}'
        os.makedirs(base_feature_path, exist_ok=True)
        return_paths[f'{feature}_samples'] = base_feature_path + f'/{feature}_hard{hardness}' + '_{pers}_{trial}.json'

        # examples
        return_paths[f'{feature}_examples_investigator'] = (
            cfg.task_config.base_path + f'/examples/investigator/{feature}/investigator_examples_{feature}_hard{hardness}.txt'
        )
        return_paths[f'{feature}_examples_user'] = (
            cfg.task_config.base_path + f'/examples/user/{feature}/user_examples_{feature}_hard{hardness}.txt'
        )

    return return_paths


def run_synthetic(cfg: Config) -> None:

    # load the personalities for the user bot
    with open(cfg.task_config.personalities_path, 'r') as f:
        user_bot_personalities = json.load(f)

    # get the bots
    investigator = get_model(cfg.task_config.investigator_bot)
    user = get_model(cfg.task_config.user_bot)
    evaluator = get_model(cfg.task_config.evaluation_model)

    for hardness in cfg.task_config.hardnesses:
    
        # get all paths at current hardness
        paths = build_paths(cfg, hardness)

        # load base prompts
        user_base_prompt = load_txt(paths['user_bot_base_prompt'])
        investigator_base_prompt = load_txt(paths['investigator_bot_base_prompt'])

        for pers, pers_dict in user_bot_personalities.items():
            
            for feature in cfg.task_config.features:
                
                # load examples
                user_examples = load_txt(paths[f'{feature}_examples_user'])
                investigator_examples = load_txt(paths[f'{feature}_examples_investigator']) 

                # build system prompts
                user_system_prompt = build_user_bot_prompt(
                    base_prompt=user_base_prompt,
                    personality=pers_dict,
                    feature=feature,
                    hardness=hardness,
                    examples=user_examples
                )
                investigator_system_prompt = build_investigator_bot_prompt(
                    base_prompt=investigator_base_prompt,
                    feature=feature,
                    examples=investigator_examples
                )

                # get which trial
                trial = get_trial(paths[f'{feature}_samples'].format(pers=pers, trial=0))
                
                # break off if we are already done with this setup
                if cfg.task_config.max_trial is not None and trial > cfg.task_config.max_trial:
                    continue
                
                try:
                    # ----- Conversation ----- #
                    conv = Conversation(
                        system_prompt=investigator_system_prompt,
                        prompts=[]
                    )
                    # investigator asks
                    initial_question = investigator.continue_conversation(conv)
                    conv.add_prompt(Prompt(role='user', intermediate=initial_question))
                    # user answers
                    conv.set_system_prompt(user_system_prompt)
                    user_answer = user.continue_conversation(conv)
                    conv.add_prompt(Prompt(role='assistant', intermediate=user_answer))
                    # investigator makes a guess
                    conv.swap_roles({'user': 'assistant', 'assistant': 'user'})
                    conv.set_system_prompt(investigator_system_prompt)
                    guess = investigator.continue_conversation(conv)

                    # ----- Evaluation of Guess ----- #
                    parsed_guesses = extract_predictions(guess)[feature]
                    model_free_eval = evaluate_guess_synthetic(
                        feature=feature,
                        ground_truth=pers_dict[feature],
                        guess=parsed_guesses,
                        evaluation_model=evaluator,
                        use_model_aid=False
                    )
                    model_aided_eval = evaluate_guess_synthetic(
                        feature=feature,
                        ground_truth=pers_dict[feature],
                        guess=parsed_guesses,
                        evaluation_model=evaluator,
                        use_model_aid=True
                    )
                    eval_dict = {'model_free_eval': model_free_eval, 'model_aided_eval': model_aided_eval}

                    sample_dict = {
                        'personality': pers_dict,
                        'feature': feature,
                        'hardness': hardness,
                        'question_asked': initial_question,
                        'response': user_answer,
                        'guess': guess,
                        'guess_correctness': eval_dict 
                    }

                    with open(paths[f'{feature}_samples'].format(pers=pers, trial=trial), 'w') as f:
                        json.dump(sample_dict, f)
                
                except Exception as e:
                    print(f'{hardness}, {pers}, {feature}: An exception occurred: {e}')
