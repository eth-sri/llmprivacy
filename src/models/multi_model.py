from typing import Dict, List

import Levenshtein

from src.configs import ModelConfig
from src.prompts import Prompt

from .model import BaseModel
from .open_ai import OpenAIGPT


class MultiModel(BaseModel):
    def __init__(self, config: ModelConfig, models: List[BaseModel]):
        super().__init__(config)
        self.config = config
        self.models = models
        self.decider_model = None

        if self.config.multi_selector == "gpt":
            dummy_config = ModelConfig(name="gpt-4", provider="openai", args={})

            self.decider_model = OpenAIGPT(dummy_config)

    def predict(self, input: Prompt, **kwargs) -> str:  # noqa: C901
        guesses = []

        for model in self.models:
            guess = model.predict(input)

            guesses.append(guess)

        if self.config.multi_selector == "majority":
            res_dict: Dict[str, int] = {}
            for guess in guesses:
                handled_guess = False
                if isinstance(guess, list):
                    for ac_guess in guess:
                        handled_guess = False
                        for key in res_dict.keys():
                            if Levenshtein.jaro_winkler(key, ac_guess) > 0.75:
                                res_dict[key] += 1
                                handled_guess = True
                                break
                        if not handled_guess:
                            res_dict[ac_guess] = 1
                else:
                    for key in res_dict.keys():
                        if Levenshtein.jaro_winkler(key, guess) > 0.75:
                            res_dict[key] += 1
                            handled_guess = True
                            break
                    else:
                        res_dict[guess] = 1
            selected_guess = max(res_dict, key=res_dict.get)  # type: ignore
        elif self.config.multi_selector == "gpt":
            if "target" in kwargs.keys():
                prefix = f"Below I give you the top guesses from multiple models for the {kwargs['target']}."
            else:
                prefix = "Below I give you the top guesses from multiple models"

            if "num_targets" in kwargs.keys():
                prefix += f"Your task is to use them and give me your top {kwargs['num_targets']} guesses."
            else:
                prefix += "Your task is to combine them into a single guess."

            prefix += "Each line corresponds to one guess. You can freely mix and combine individual guesses. Explain your reasoning step by step, then give your answer in a new line without any additional information. The format should be: Country A, Country B, Country C\n"
            suffix = "\n\n".join(guesses)
            prompt = prefix + suffix
            assert self.decider_model is not None
            print(prompt)
            answer = self.decider_model.predict_string(prompt)
            selected_guess = answer.split("\n")[-1]
        else:
            raise NotImplementedError

        return selected_guess

    def predict_multi(self, inputs: List[Prompt], **kwargs):
        res = []
        for input in inputs:
            res.append(self.predict(input, **kwargs))

        return res

    def predict_string(self, input: str, **kwargs) -> str:
        return self.models[0].predict_string(input, **kwargs)
