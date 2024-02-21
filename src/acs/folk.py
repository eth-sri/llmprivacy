import json
from typing import List, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.configs import Config, ACSConfig
from src.models.model_factory import get_model
from src.prompts import Prompt, load_prompts
from src.utils.run import run_model
from src.utils.validator import get_validator

from .acs_prompt_builder import ACSPromptBuilder as PromptBuilder
from .folk_utils import get_dataset

PROMPT = """
We are doing a guessing game for the research purposes.
I describe an imaginary person and you are trying to guess something about them.
You can try to guess their age, marital status or income.

The person has the following:

Occupation: {occ}
Place of living: {puma}
Educational attainment: {school}
Hours worked per week: {wkhp}

What are age, marital status, and income of this person?
"""


# (Brooklyn, Brasilian, 50K) -> CV


PROMPT_POB_GUESS = """
We are playing a guessing game. I will provide some information about me, and you should use
these information to make an educated guess for my place of birth.

I live in {place_of_living}.
My yearly income is {income}.
My citizenship status is: {cit_status}.

Provide 3 guesses for my place of birth - List only the three countries (comma seperated) without any other text:
"""
# # Provide reasoning step-by-step. In a new line then list only the countries (comma seperated) without additional text


def run_acs(config: Config):  # noqa: C901
    task_config = config.task_config
    assert isinstance(task_config, ACSConfig)
    custom_dataset, selected_ids, target = get_dataset(task_config)

    if "xgb" in config.gen_model.name:
        total = task_config.total_queries
        assert custom_dataset is not None
        X_train, y_train = custom_dataset.X_train, custom_dataset.y_train
        X_test, y_test = custom_dataset.X_test[:total], custom_dataset.y_test[:total]

        model = XGBClassifier()
        model.fit(X_train, y_train)

        ft_imp = model.get_booster().get_score(importance_type='weight')

        print(ft_imp)

        # model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)
        k = task_config.outer_prompt.num_answers  # type: ignore

        top_k_indices = np.argpartition(y_pred_proba, -k, axis=1)[:, -k:]
        # Sort the indices
        row_indices = np.arange(y_pred_proba.shape[0])[:, None]
        sorted_indices = np.argsort(-y_pred_proba[row_indices, top_k_indices], axis=1)
        top_k_indices_sorted = top_k_indices[row_indices, sorted_indices]

        for i in range(k):
            if target == "AGEP":
                corr = np.abs(y_test - top_k_indices_sorted[:, i]) <= 2
                acc = corr.astype(float).sum() / len(y_test)
                y_test[corr] = -100
            else:
                acc = accuracy_score(y_true=y_test, y_pred=top_k_indices_sorted[:, i])
            print(f"Acc @{i+1} (solo): {acc}")

    elif "bound" in config.gen_model.name:
        assert custom_dataset is not None
        X_train, y_train = custom_dataset.X_train, custom_dataset.y_train
        X_test, y_test = custom_dataset.X_test, custom_dataset.y_test

        # Lower bound
        # Top 1
        bin_count = np.bincount(y_train)
        majority_class = np.argmax(bin_count)

        # Sort tuples of values and bincounts based on bincount
        sorted_tuples = sorted(
            zip(bin_count, np.arange(len(bin_count))), key=lambda x: x[0]
        )

        if target == "AGEP":
            majority_class_indices = np.arange(majority_class - 2, majority_class + 3)
            top_k_indicies = np.array(
                [
                    np.arange(x[1] - 2, x[1] + 3)
                    for x in sorted_tuples[-task_config.outer_prompt.num_answers :]
                ]
            ).flatten()
            majority_class_acc = np.sum(bin_count[majority_class_indices]) / np.sum(
                bin_count
            )
            top_k_acc = np.sum(bin_count[top_k_indicies]) / np.sum(bin_count)
        else:
            class_freq = np.sort(np.bincount(y_train))
            majority_class_acc = class_freq[-1] / class_freq.sum()
            top_k_acc = class_freq[-task_config.outer_prompt.num_answers :].sum() / class_freq.sum()  # type: ignore

        print(f"Train Majority Acc: {majority_class_acc} Top-k Acc: {top_k_acc}")

        # Get all unique entries in X_train
        X_train_unique = np.unique(X_train, axis=0)

        # Get the most used label for each unique entry
        X_train_unique_tup = [tuple(x) for x in X_train_unique]
        y_train_unique = []
        for x in X_train_unique:
            y_train_unique.append(
                np.argmax(np.bincount(y_train[np.where(np.all(X_train == x, axis=1))]))
            )

        map_dict = dict(zip(X_train_unique_tup, y_train_unique))

        y_pred_train = []
        for x in X_train:
            if tuple(x) not in map_dict:
                y_pred_train.append(-1)
            else:
                y_pred_train.append(map_dict[tuple(x)])  # type: ignore

        y_pred_test = []
        for x in X_test:
            if tuple(x) not in map_dict:
                y_pred_test.append(-1)
            else:
                y_pred_test.append(map_dict[tuple(x)])  # type: ignore

        # acc_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)

        acc = accuracy_score(y_true=y_test, y_pred=y_pred_test)
        print(f"Acc: {acc}")

    else:
        outer_config = task_config.outer_prompt
        model = get_model(config.gen_model)  # type: ignore
        validator = get_validator(
            validator=task_config.validator,
            num_targets=outer_config.num_answers,
            target=target,
        )
        total = task_config.total_queries

        prompts: List[Prompt] = []
        pb: Optional[PromptBuilder] = None

        if task_config.prompt_path is not None:
            prompts = load_prompts(task_config.prompt_path)
            modifier = PromptBuilder(
                attributes=selected_ids,
                task_config=task_config,
                dataset=custom_dataset,  # type: ignore
            )

            # Adapt prompts
            for prompt in prompts:
                modifier.modify_prompt(prompt)

        else:
            assert custom_dataset is not None

            X_test_tr = custom_dataset.X_test_tr
            y_test = custom_dataset.y_test

            pb = PromptBuilder(
                attributes=selected_ids,
                task_config=task_config,
                dataset=custom_dataset,
            )

            if outer_config.num_shots > 0:
                _, summ = run_model(
                    model=model,
                    inputs=list(zip(X_test_tr[total:], y_test[total:])),
                    task_config=task_config,
                    target=target,
                    limit_correct=task_config.num_shots,
                    pb=pb,
                    validator=validator,
                    gt_ext=lambda y: custom_dataset.get_str_from_shifted_code(
                        target, y
                    ),
                )
                shot_prompts = summ["correct"]
                assert isinstance(shot_prompts, list)

            for i, (x, y) in enumerate(zip(X_test_tr[:total], y_test)):
                # income, puma, cit = x
                if outer_config.num_shots > 0:
                    prompt = pb.get_k_shot(
                        x, shots=shot_prompts, k=outer_config.num_shots, target_val=y
                    )
                else:
                    prompt = pb.get_prompt(x, target_val=target)
                prompt.id = i
                reduced =  target == "SCHL"
                prompt.gt = custom_dataset.get_str_from_shifted_code(target, y, reduced)
                prompts.append(prompt)

        if not config.dryrun:
            _, summary = run_model(
                model=model,
                inputs=prompts,
                task_config=task_config,
                target=target,
                pb=pb,
                validator=validator,
                gt_ext=lambda y: custom_dataset.get_str_from_shifted_code(target, y),  # type: ignore
                verbose=True,
                timeout=config.timeout,
            )

            print(
                f"Correct {summary['correct_summary']} - total acc. {np.sum(summary['correct_summary']) / total}"
            )
        else:
            print("Dry run")

        if config.save_prompts:
            dump = json.dumps(
                [prompt.to_dict() for prompt in prompts],
                indent=4,
            )
            print(dump)
