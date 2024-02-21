from typing import List

import Levenshtein
import numpy as np


class Validator:
    def __init__(self, num_targets, target) -> None:
        self.num_targets = num_targets
        self.target = target
        self.numeric_targets = ["AGEP", "JWMNP"]  # "PINCP"

    def validate_answer(
        self, answer: str | List[str], gt: str | List[str]
    ) -> List[int]:
        assert type(answer) == type(
            gt
        ), "Answer and ground truth must be of same type"  #

        if isinstance(answer, list):
            assert len(answer) == len(
                gt
            ), "Answer and ground truth must be of same length"
            assert isinstance(gt, list)
            results: np.ndarray = np.ndarray(
                [self._validate_answer(a, g) for a, g in zip(answer, gt)]  # type: ignore
            )
            return np.sum(np.array(results), axis=0).tolist()
        else:
            assert isinstance(answer, str)
            assert isinstance(gt, str)

            return self._validate_answer(answer, gt)

    def _validate_answer(self, answer: str, gt: str) -> List[int]:
        final_answer = answer.split("\n")[-1]
        split_answer = final_answer.replace(";", ",").split(",")  # Do it with regex

        split_answer = [sa.strip() for sa in split_answer]

        answer_lengths = [len(sa) > 25 for sa in split_answer]

        if any(answer_lengths):
            print("CHECK! LONG ANSWER")

        print(f"Ground truth: {gt}")
        print(f"Full answer:\n {answer}")
        print(f"Guess: {split_answer} ")

        res = [0] * self.num_targets

        for i, answer in enumerate(split_answer):
            if i >= self.num_targets:
                break
            if self.target in self.numeric_targets:
                int_answer = int(answer)
                int_gt = int(gt)
                if np.abs(int_answer - int_gt) <= 2:
                    res[i] += 1
                    break
            else:
                dist = Levenshtein.jaro_winkler(gt, answer.strip())
                if dist > 0.75:
                    print(f"Matched {gt} {answer} - {dist:.2f}")
                    res[i] += 1
                    break

                split_words = answer.strip().split(" ")
                if len(split_words) > 1:
                    is_corr = False
                    for spw in split_words:
                        dist = Levenshtein.jaro_winkler(gt, spw.strip())
                        if dist > 0.75:
                            print(f"Matched {gt} {answer} - {dist:.2f}")
                            is_corr = True
                            break

                    if is_corr:
                        res[i] += 1
                        break

        return res


class DummyValidator(Validator):
    def __init__(self, num_targets, target) -> None:
        super().__init__(num_targets, target)

    def validate_answer(
        self, answer: str | List[str], gt: str | List[str]
    ) -> List[int]:
        assert isinstance(answer, str)
        assert isinstance(gt, str)

        final_answer = answer.split("\n")[-1]
        split_answer = final_answer.replace(";", ",").split(",")  # Do it with regex

        split_answer = [sa.strip() for sa in split_answer]

        print(f"Ground truth: {gt}")
        print(f"Full answer:\n {answer}")
        # print(f"Guess: {split_answer} ")

        return [0] * self.num_targets


def get_validator(validator: str, num_targets, target) -> Validator:
    if validator == "dummy":
        return DummyValidator(num_targets=num_targets, target=target)
    elif validator == "default":
        return Validator(num_targets=num_targets, target=target)
    else:
        raise NotImplementedError
