from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
)
from src.configs import REDDITConfig
from src.models.model_factory import get_model
from src.reddit.reddit_utils import load_data
from src.reddit.reddit import create_prompts


if __name__ == "__main__":

    ###
    # Very basic demo creating one request on one piece of synthetic data
    ###

    data_idx = 23

    ### Environment setup
    cfg = read_config_from_yaml("configs/reddit/running/reddit_synthetic_gpt4.yaml")

    seed_everything(cfg.seed)
    # If using OpenAI API or other external services -> Fill in fields in credentials.py (and exclude from GIT!!!)
    set_credentials(cfg)

    ### Run the task

    model = get_model(cfg.gen_model)

    assert isinstance(cfg.task_config, REDDITConfig)
    profiles = load_data(cfg.task_config.path)

    # Create prompt
    prompt = create_prompts(profiles[data_idx], cfg.task_config)[0]

    # Run the model
    max_workers = 1
    timeout = 40

    print(prompt)

    results = model.predict_multi([prompt], max_workers=max_workers, timeout=timeout)

    print("=============\nAnswer\n=============\n")

    for res in results:
        # Answer
        print(res[1])
