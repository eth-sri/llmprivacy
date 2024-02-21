import json
from typing import List

from .prompt import Prompt


def load_prompts(path: str) -> List[Prompt]:
    try:
        with open(path, "r") as f:
            prompt_file = f.readlines()

            start_line = [
                i for i, line in enumerate(prompt_file) if line.startswith("[")
            ][0]
            end_line = [
                i for i, line in enumerate(prompt_file) if line.startswith("]")
            ][0]

            assert start_line < end_line

            prompt_file = prompt_file[start_line : end_line + 1]
            load_prompt = json.loads("".join(prompt_file))
            prompts = [Prompt.from_dict(p) for p in load_prompt]

    except FileNotFoundError as e:
        print(f"Prompt file {path} not found")
        raise e
    except IndexError as e:
        print(f"Prompt file {path} is malformed")
        raise e

    return prompts
