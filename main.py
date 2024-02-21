import argparse
import sys
from src.utils.initialization import (
    read_config_from_yaml,
    seed_everything,
    set_credentials,
    get_out_file,
)
from src.configs import Task
from src.acs.folk import run_acs
from src.pan.pan import run_pan
from src.reddit.reddit import run_reddit
from src.chat.run_chat import run_chat
from src.chat.eval_chat import run_eval_chat
from src.synthetic.synthetic import run_synthetic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/acs_config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials(cfg)

    f, path = get_out_file(cfg)

    try:
        print(cfg)
        if cfg.task == Task.ACS:
            run_acs(cfg)
        elif cfg.task == Task.PAN:
            run_pan(cfg)
        elif cfg.task == Task.REDDIT:
            run_reddit(cfg)
        elif cfg.task == Task.CHAT:
            run_chat(cfg)
        elif cfg.task == Task.CHAT_EVAL:
            run_eval_chat(cfg)
        elif cfg.task == Task.SYNTHETIC:
            run_synthetic(cfg)
        else:
            raise NotImplementedError(f"Task {cfg.task} not implemented")

    except ValueError as e:
        sys.stderr.write(f"Error: {e}")
    finally:
        if cfg.store:
            f.close()
