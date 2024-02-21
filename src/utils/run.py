import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.configs import ACSConfig, PANConfig
from src.models import BaseModel
from src.prompts import Prompt
from src.utils.validator import Validator


def translate_target(target: str) -> str:
    if target == "POBP":
        return "Birthplace of a person"
    elif target == "AGEP":
        return "Age of a person"
    else:
        return target


def run_model(
    model: BaseModel,
    inputs: List[Prompt] | List[Tuple[str, str]],
    task_config: ACSConfig | PANConfig,
    target: str,
    validator: Validator,
    pb: Optional[Any] = None,
    gt_ext: Callable[[Any], str] = lambda x: str(x),
    limit_correct: int = 0,
    timeout: float = 0.5,
    verbose: bool = False,
) -> Tuple[List[Prompt], Dict[str, Any]]:
    correct: List[Prompt] = []

    if isinstance(task_config, ACSConfig):
        num_answers = task_config.outer_prompt.num_answers
        correct_ctr = [0] * num_answers
    else:
        correct_ctr = [0] * 1
        num_answers = 1

    # Adapt target here
    # target = translate_target(target)

    results = model.predict_multi(
        inputs,
        target=target,
        num_targets=num_answers,
    )

    for i, result in enumerate(results):
        prompt, guess = result
        print(f"============= QUERY {i} =============")
        print(prompt.get_prompt())
        sys.stdout.flush()

        # Remove all blank lines in the end
        guess = guess.strip()

        prompt.answer = guess

        curr_res = validator.validate_answer(guess, gt=prompt.gt)
        if sum(curr_res) > 0:
            correct.append(prompt)
            for j in range(len(curr_res)):
                correct_ctr[j] += curr_res[j]

        if limit_correct > 0 and len(correct) >= limit_correct:
            break

    return results, {
        "correct_summary": correct_ctr,
        "correct": correct,
        "total": len(inputs),
    }
