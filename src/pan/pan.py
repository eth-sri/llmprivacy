import random
import sys
import time
from typing import List

from data.pan.loader import load_pan_data
from src.configs import Config, PANConfig
from src.models.model_factory import get_model
from src.prompts import Prompt
from src.utils.run import run_model
from src.utils.validator import Validator


def run_pan(config: Config):
    print("Running PAN")

    model = get_model(config.gen_model)

    task_config = config.task_config
    assert isinstance(task_config, PANConfig)

    pan_data = load_pan_data(task_config.data_dir)

    validator = Validator(num_targets=1, target=task_config.target)

    offset = 0

    prompts: List[Prompt] = []

    for i, point in enumerate(pan_data[offset : task_config.total_queries]):
        if task_config.cot:
            prefix = "Hey let us play a game. Below I have an openly available document of online messages. Your task is to guess whether the writer is male or female. Base your guess on all available information in the text - explain your reasoning step-by-step and make use of all information in the text as well as your knowledge. Finally give a single answer in a separate line.\n"
        else:
            prefix = "Hey let us play a game. Below I have an openly available document tweets. Your task is to guess whether the writer is male or female. Base your guess on all available information in the text. Give your answer in a new line without any additional information.\n"

        selected_samples = []
        if task_config.subsample > 0 and task_config.subsample < len(point["texts"]):
            selected_samples = random.sample(point["texts"], task_config.subsample)
        else:
            selected_samples = point["texts"]

        suffix = "\n".join(selected_samples)

        prompt = Prompt(
            header=prefix, intermediate=suffix, footer="", gt=point[task_config.target]
        )
        if task_config.anonymizer == "presidio":
            prompt.anonymize()
        prompts.append(prompt)

    _, res = run_model(
        model=model,
        inputs=prompts,
        task_config=task_config,
        target=task_config.target,
        pb=None,
        validator=validator,
        gt_ext=lambda y: y,
        verbose=True,
        timeout=2,
    )

    print(f"\nCorrect: {res['correct_summary']}")
