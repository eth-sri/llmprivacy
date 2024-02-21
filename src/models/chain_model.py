import re
from typing import Any, Dict, List

from src.configs import ModelConfig
from src.prompts import Prompt
from src.utils.string_utils import replace_parentheses

from .model import BaseModel

# Format: 0.header N.footer


class ChainModel(BaseModel):
    def __init__(self, config: ModelConfig, models: List[BaseModel]):
        super().__init__(config)
        self.config = config
        self.models = models

    def predict(self, input: Prompt, **kwargs) -> str:  # noqa: C901
        curr_prompt = input
        prompts: Dict[int | str, Any] = {0: curr_prompt, "prev": curr_prompt}

        for j, model in enumerate(self.models):
            prompts["self"] = model.config.prompt_template

            new_header = ""
            new_footer = ""
            new_interm = ""

            if "header" in model.config.prompt_template:
                new_header = fill_template_string(
                    model.config.prompt_template["header"], prompts  # type: ignore
                )
                curr_prompt.header = new_header
            if "footer" in model.config.prompt_template:
                new_footer = fill_template_string(
                    model.config.prompt_template["footer"], prompts  # type: ignore
                )
                curr_prompt.footer = new_footer
            if "template" in model.config.prompt_template:
                new_interm = fill_template_string(
                    model.config.prompt_template["template"], prompts  # type: ignore
                )
                curr_prompt.intermediate = new_interm

            output = model.predict(curr_prompt)
            curr_prompt.answer = output
            prompts[j + 1] = curr_prompt
            prompts["prev"] = curr_prompt

            print(f"==Model {j}\n=Input:\n {curr_prompt.get_prompt()}")
            print(f"=Output:\n {output}")

            curr_prompt = curr_prompt.get_copy()

        assert curr_prompt.answer is not None
        return curr_prompt.answer

    def predict_multi(self, inputs: List[Prompt], **kwargs):
        res = []
        for input in inputs:
            res.append(self.predict(input, **kwargs))

        return res

    def predict_string(self, input: str, **kwargs) -> str:
        return self.models[0].predict_string(input, **kwargs)


def fill_template_string(template: str, fill_dict: Dict[str, Any]) -> str:
    extracted_locations = re.findall("\{.*?\}", template)  # noqa: W605
    to_insert = replace_parentheses(template)

    for i, loc in enumerate(extracted_locations):
        index, identifier = loc[1:-1].split(".")

        if index not in ["self", "prev"]:
            try:
                index = int(index)
            except ValueError:
                raise ValueError(f"Index {index} is not a valid index.")
        selected_elem = fill_dict[index]

        selected_val = ""
        if isinstance(selected_elem, Prompt):
            selected_val = getattr(selected_elem, identifier)
        elif isinstance(selected_elem, dict):
            selected_val = selected_elem[identifier]
        else:
            raise NotImplementedError

        to_insert = to_insert.replace(f"<{i+1}>", selected_val)

    return to_insert
