import os
import random
import sys
from typing import TextIO, Tuple, no_type_check, Optional

import numpy as np
import openai
import yaml
from pydantic import ValidationError

from src.configs import Config

from .string_utils import string_hash


class SafeOpen:
    def __init__(self, path: str, mode: str = "a", ask: bool = False):
        self.path = path

        if ask and os.path.exists(path):
            if input("File already exists. Overwrite? (y/n)") != "y":
                raise Exception("File already exists")
        self.mode = "a+" if os.path.exists(path) else "w+"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.file = None
        self.lines = []

    def __enter__(self):
        self.file = open(self.path, self.mode)
        self.file.seek(0)  # move the cursor to the beginning of the file
        self.lines = self.file.readlines()
        # Remove last lines if empty
        while len(self.lines) > 0 and self.lines[-1] == "":
            self.lines.pop()
        return self

    def flush(self):
        self.file.flush()

    def write(self, content):
        self.file.write(content)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


def read_config_from_yaml(path) -> Config:
    with open(path, "r") as stream:
        try:
            yaml_obj = yaml.safe_load(stream)
            print(yaml_obj)
            cfg = Config(**yaml_obj)
            return cfg
        except (yaml.YAMLError, ValidationError) as exc:
            print(exc)
            raise exc


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.set_num_threads(1)


def set_credentials(model_cfg: Optional[Config] = None) -> None:
    if model_cfg is None or model_cfg.gen_model.provider == "openai":
        from credentials import openai_api_key, openai_org

        openai.organization = openai_org
        openai.api_key = openai_api_key
    elif model_cfg.gen_model.provider == "azure":
        from credentials import azure_endpoint, azure_key, azure_api_version

        openai.api_type = "azure"
        openai.api_base = azure_endpoint
        openai.api_key = azure_key
        openai.api_version = azure_api_version


@no_type_check
def get_out_file(cfg: Config) -> Tuple[TextIO, str]:
    file_path = cfg.get_out_path(cfg.task_config.get_filename())

    if not cfg.store:
        return sys.stdout, ""

    if len(file_path) > 255:
        file_path = file_path.split("/")
        file_name = file_path[-1]
        file_name_hash = string_hash(file_name)
        file_path = "/".join(file_path[:-1]) + "/hash_" + str(file_name_hash) + ".txt"

    ctr = 1
    while os.path.exists(file_path):
        with open(file_path, "r") as fp:
            num_lines = len(fp.readlines())

        if num_lines >= 20:
            file_path = file_path.split("/")
            file_name = file_path[-1]

            ext = file_name.split(".")[-1]
            v_counter = file_name.split("_")[-1].split(".")[0]
            if v_counter.isdigit():
                ext_len = len(ext) + len(v_counter) + 2
                file_name = file_name[:-ext_len]
            else:
                file_name = file_name[: -(len(ext) + 1)]

            file_path = (
                "/".join(file_path[:-1]) + "/" + file_name + "_" + str(ctr) + ".txt"
            )
            ctr += 1
        else:
            break

    if cfg.store:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f = open(file_path, "w")
        sys.stdout = f

    return f, file_path
