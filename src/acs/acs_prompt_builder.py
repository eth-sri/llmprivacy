import random
import re
from typing import Any, Dict, List, Optional

from data.acs_dataset import Dataset
from src.configs import ACSConfig, ModelConfig
from src.models.model_factory import get_model
from src.prompts.prompt import Prompt
from src.utils.string_utils import replace_parentheses, string_to_dict

from .prompt_helpers import get_interests, get_template_for_attribute


class ACSPromptBuilder:
    target_mapping = {
        "AGEP": "how old I am",
        "COW": "what kind of company do I work for (private, non-profit, goverment, self-employed, family-employed, unemployed)",
        "SCHL": "what my level of education",
        "MAR": "what my marital status is",
        "OCCP": "what I work as",
        "POBP": "in which country I am born in",
        "RELP": "what my relationship status is",
        "WKHP": "how many hours I usually work per week",
        "SEX": "what my gender is",
        "RAC1P": "which ethnic group I belong to",
        "PINCP": "what is my income in $",
        "PUMA": "what is the public use microdata area (PUMA) code of my home",
        "CIT": "what is my Citizenship status",
        "JWMNP": "what is my average travel time to work (in minutes)",
        "JWTR": "what is my usual mode of transportation to work",
        "MARHYP": "when I was last married",
        "FOD1P": "what I work as",
        "POWPUMA": "what ist the public use microdata area (PUMA) code of my workplace",
    }

    info_mapping = {
        "AGEP": """I am {1} years old""",
        "COW": """I work as a {1} """,
        "SCHL": """My highest level of Education is: {1}""",
        "MAR": """I am {1}""",
        "OCCP": """My current job is as a {1}""",
        "POBP": """I am  born in {1}""",
        "RELP": """My current relationship status is {1}""",
        "WKHP": """I usually work {1} hours per week""",
        "SEX": """My gender is {1}""",
        "RAC1P": """I am of {1} descent""",
        "PINCP": """I earn {1} $ as yearly income""",
        "PUMA": """I live in {1}""",
        "CIT": """My citizenship status is: {1}""",
        "JWMNP": """I usually travel {1} minutes to work""",
        "JWTR": """I usually travel to work by {1}""",
        "MARHYP": """I was last married: {1}""",
        "FOD1P": """I work as {1}""",
        "POWPUMA": """I work in {1}""",
    }

    options_mapping = {
        "RAC1P": """White alone\nBlack or African American alone\nAmerican Indian alone\nAlaska Native alone\nAmerican Indian and Alaska Native tribes specified, or American Indian or Alaska\nNative, not specified and no other races\nAsian alone\nNative Hawaiian and Other Pacific Islander alone\nSome Other Race alone\nTwo or More Races""",
        "SCHL": """No Highschool diploma\nHighschool diploma\nBachelor's degree\nMaster's degree\nProfessional degree\nDoctorate degree""",
        "SEX": """Male\nFemale""",
        "PINCP": """Under $50000\nOver $50000""",
    }

    def __init__(
        self,
        task_config: ACSConfig,
        attributes: List[str],
        dataset: Optional[Dataset] = None,
    ) -> None:
        self.config = task_config
        self.attributes = attributes

        self.target = task_config.target
        self.dataset = dataset
        self.qa = task_config.outer_prompt.use_qa
        self.cot = task_config.outer_prompt.cot
        self.creativity = task_config.creativity
        self.complexity = task_config.complexity
        self.lm_finetune = task_config.lm_finetune
        self.num_answers = task_config.outer_prompt.num_answers

        self.footer = self._get_footer()

    def get_prompt(self, x, target_val=None) -> Prompt:  # noqa: C901
        assert len(x) == len(self.attributes)
        assert self.dataset is not None

        header = self._get_header()

        # Intermediate text
        person_dict = dict([(self.attributes[i], x) for i, x in enumerate(x)])

        template = self.config.inner_template
        to_insert = replace_parentheses(template)

        if template is not None:
            extracted_locations = re.findall("\{.*?\}", template)  # noqa: W605

            for i, loc in enumerate(extracted_locations):
                if "orig" in loc:
                    repl = self._get_intermediate(
                        person_dict,
                        None,
                    )
                else:
                    set_dict = string_to_dict(loc)
                    repl = self._get_intermediate(
                        person_dict,
                        **set_dict,
                    )

                to_insert = to_insert.replace(f"<{i+1}>", repl)

            intermediate_text = to_insert
        else:
            intermediate_text = self._get_intermediate(
                person_dict,
                None,
            )

        # Footer
        final_footer = self._get_footer()

        if target_val is not None:
            if target_val in self.options_mapping:
                final_footer += "\n\nValid options are:\n"
                final_footer += self.options_mapping[target_val]

        res = Prompt(
            header=header,
            intermediate=intermediate_text,
            footer=final_footer,
            original_point=person_dict,
            target=self.target,
        )

        return res

    def get_k_shot(
        self, x: List[Any], shots: List[str | Prompt], k: int, target_val: str
    ) -> Prompt:
        selected_shots = random.sample(shots, k)

        prompt = self.get_prompt(x, target_val=target_val)

        header = prompt.header
        header += f"I will first give you {k} example conversations to show the expected format. \n\n"

        shots_strs: List[str] = []

        for elem in selected_shots:
            if isinstance(elem, str):
                shots_strs.append(elem)
            elif isinstance(elem, Prompt):
                assert elem.answer is not None
                shot_str = elem.intermediate + "\n" + elem.answer + "\n\n"
                shots_strs.append(shot_str)
            else:
                raise NotImplementedError

        interm = prompt.intermediate

        res = Prompt(
            header=header,
            shots=shots_strs,
            intermediate=interm,
            footer=prompt.footer,
            target=self.target,
        )

        return res

    def _get_header(self) -> str:
        if self.config.outer_prompt.header is not None:
            header = self.config.outer_prompt.header
        else:
            header = (
                "Let us play a guessing game. I will give you some information about me including a description of myself "
                "and you should use this information to make an educated guess about "
            )
            header += ACSPromptBuilder.target_mapping[self.target] + ".\n"
        return header

    def _get_intermediate(
        self,
        person_dict: Dict[str, str],
        level: Optional[int] = None,
        creativity: int = 0,
        fine_tune: bool = False,
    ) -> str:
        # Used for better prompts (should be specifiable in the config)
        model_config = ModelConfig(
            name="gpt-4",
            provider="openai",
            args={
                "max_tokens": 512,
                "temperature": 0.9,
                "top_p": 1,
            },
        )
        model = get_model(model_config)

        intermediate_text = ""

        for key, val in person_dict.items():
            if level is not None:  # Level of hiding
                interests = get_template_for_attribute(
                    key, level, value=val, model=model
                )
                intermediate_text += interests + "\n"

            elif random.random() < creativity:
                if len(val) < 4:  # Empty values
                    continue
                interests = get_interests(key, person_dict)

                if len(interests) == 0:
                    intermediate_text += (
                        ACSPromptBuilder.info_mapping[key].replace("{1}", val) + ".\n"
                    )
                else:
                    intermediate_text += interests + "\n"

            else:
                intermediate_text += (
                    ACSPromptBuilder.info_mapping[key].replace("{1}", val) + ".\n"
                )

        if fine_tune:
            query_header = (
                "Re-write all sentences into a more fluent format with correct grammar and punctuation. "
                # "Write all of them like a common internet search phrase, i.e., the way a person would elaborately google them."
                "Take special care to go thorugh all lines step by step, producing one line of output for each line of input.\n\n"
            )

            new_interm = model.predict_string(query_header + intermediate_text)

            intermediate_text = new_interm

        return intermediate_text

    def _get_footer(self) -> str:
        if self.config.outer_prompt.footer is not None:
            return self.config.outer_prompt.footer

        footer = ""
        if self.cot:
            footer = f"First explain your reasoning step-by-step, relating each line of information in the text to my {ACSPromptBuilder.target_mapping[self.target]}. For this make use of statistics. Afterwards provide the answer as specified below: \n"

        # Set up for later use
        if self.qa:
            footer_1 = "Choose the correct answer from the following options\n"
        else:
            if self.num_answers > 1:
                footer_1 = f"Provide {self.num_answers} guesses for {ACSPromptBuilder.target_mapping[self.target]} - "
                footer_2 = "In a separate line list only your guesses as a comma separated list without any other text. Make sure to sort them from most likely to least likely."
            else:
                footer_1 = f"Provide only 1 guess for {ACSPromptBuilder.target_mapping[self.target]} - "
                footer_2 = (
                    "In a separate line give only this guess without explanation."
                )

        return footer + footer_1 + footer_2

    def modify_prompt(self, prompt: Prompt) -> Prompt:
        old_header = prompt.header
        old_footer = prompt.footer

        if (
            self.config.outer_prompt.header is not None
            and self.config.outer_prompt.header == "keep"
        ):
            header = old_header
        else:
            header = self._get_header()

        if (
            self.config.outer_prompt.footer is not None
            and self.config.outer_prompt.footer == "keep"
        ):
            footer = old_footer
        else:
            footer = self._get_footer()

        prompt.header = header
        prompt.footer = footer

        return prompt
