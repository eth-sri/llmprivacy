import json
from src.configs import Config
from src.models.model_factory import get_model
from src.prompts import Prompt, Conversation
from .user_bot_system_prompt_builder import build_user_bot_system_prompt
from .chat_parser import check_if_guess, parse_investigator_response


def run_chat(cfg: Config) -> None:

    # assign the roles and the chat run
    valid_roles = ['investigator', 'user']
    assert cfg.task_config.chat_starter in valid_roles, f'{cfg.task_config.chat_starter} is an invalid \
        choice for the conversation starter bot. Please select from {valid_roles}.'
    if cfg.task_config.chat_starter == 'user':
        valid_roles.reverse()  # bit hacky
    chat_bot_rounds = [valid_roles[round%2] for round in range(cfg.task_config.n_rounds)]

    # load the models
    models = {}
    models['investigator'] = get_model(cfg.task_config.investigator_bot)
    models['user'] = get_model(cfg.task_config.user_bot)

    # load the system prompts
    with open(cfg.task_config.investigator_bot_system_prompt_path, 'r') as f:
        investigator_bot_system_prompt = f.read()
    with open(cfg.task_config.user_bot_system_prompt_path, 'r') as f:
        user_bot_system_prompt_skeleton = f.read()

    # load the personalities for the user bot
    with open(cfg.task_config.user_bot_personalities_path, 'r') as f:
        user_bot_personalities = json.load(f)

    # in case a specific personality is given, reduce to that
    if cfg.task_config.user_bot_personality is not None:
        user_bot_personalities = (
            {f'pers{cfg.task_config.user_bot_personality}': user_bot_personalities[f'pers{cfg.task_config.user_bot_personality}']}
        )
    
    # iterate over the personalities and simualate the conversation
    for pers, pers_profile in user_bot_personalities.items():

        try:

            pers_user_bot_system_prompt = build_user_bot_system_prompt(
                prompt_skeleton=user_bot_system_prompt_skeleton,
                profile=pers_profile
            )

            # init convos
            conv_user = Conversation(
                system_prompt=pers_user_bot_system_prompt,
                prompts=[]
            )
            conv_investigator = Conversation(
                system_prompt=investigator_bot_system_prompt,
                prompts=[]
            )

            convs = {
                'investigator': conv_investigator,
                'user': conv_user
            }

            # run the convo
            for bot_name in chat_bot_rounds:
                
                response = models[bot_name].continue_conversation(convs[bot_name])
                
                # always record with the investigator
                prompt_to_investigator = Prompt(
                    role='user' if bot_name == 'user' else 'assistant',
                    intermediate=response
                )
                conv_investigator.add_prompt(prompt_to_investigator)

                # if a guess has been made, we end the conversation
                if check_if_guess(response):
                    break
                
                if cfg.task_config.mode == 'reasoning' and bot_name == 'investigator':
                    # process the investigator's response before giving it to the user
                    _, response_to_user, _ = parse_investigator_response(response)
                    prompt_to_user = Prompt(
                        role='user' if bot_name == 'investigator' else 'assistant',
                        intermediate=response_to_user
                    )
                else:
                    prompt_to_user = Prompt(
                        role='user' if bot_name == 'investigator' else 'assistant',
                        intermediate=response
                    )

                conv_user.add_prompt(prompt_to_user)

            # print the conversation
            print('\n\n')
            print(f'# ------ Conversation with {pers} ------ #')
            print(pers_profile)
            print(f'Conversation Start')
            for bot_name, prompt in zip(chat_bot_rounds, conv_investigator.prompts):
                print('\n')
                print(bot_name.upper() + ':')
                print(prompt.get_prompt().strip())
        
        except Exception as e:
            print('\n\n')
            print(f'# ------ Conversation with {pers} ------ #')
            print(f'{pers}: An exception occurred')
            print(e)
