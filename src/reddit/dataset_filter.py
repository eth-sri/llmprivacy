import argparse
from typing import List, Any, Dict
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.reddit.reddit_utils import load_data  # noqa: E402
from src.reddit.reddit_types import Profile  # noqa: E402
from src.utils.initialization import SafeOpen  # noqa: E402


def get_model_attr_map(profile: Profile) -> Dict[str, Dict[str, List[int]]]:
    model_attr_map: Dict[str, Dict[str, List[int]]] = {}

    for model, reviewers in profile.evaluations.items():
        for reviewer, attributes in reviewers.items():
            for attribute, values in attributes.items():
                if model not in model_attr_map:
                    model_attr_map[model] = {}

                if attribute not in model_attr_map[model]:
                    model_attr_map[model][attribute] = values
                else:
                    assert False, "Duplicte attribute"

    return model_attr_map


def filter_ds(
    path: str,
    outpath: str,
    reference: List[str],
    pii_types: List[str],
    no_human: bool,
    min_hardness: int,
    min_certainty: int,
    tolerance: int,
) -> None:
    """Filters the dataset based on the reference and the PII types.

    Args:
        path (str): Path to the dataset
        outpath (str): Path to the output dataset
        reference (List[str]): List of model names that should be used as reference
        pii_types (List[str]): List of PII types that should be looked at
        allow_human (bool): Whether to also return labels where the model is in accordance with the human labelers
        hardness (int): Minimum hardness of the PII
        certainty (int): Minimum certainty of the PII
        tolerance (int): Tolerance for number of other models that can be correct
    """

    profiles = load_data(path)

    filtered_profiles: List[Profile] = []

    for profile in profiles:
        model_attr_map = get_model_attr_map(profile)

        # Check if we our target models have the attributes correct
        found_attributes = []

        for attr in pii_types:
            models_consistent = True
            for model in reference:
                if model not in model_attr_map:
                    models_consistent = False
                    break
                if attr not in model_attr_map[model]:
                    models_consistent = False
                    break

                if len(model_attr_map[model][attr]) > 0:
                    predicts_correct = max(model_attr_map[model][attr]) == 1
                else:
                    predicts_correct = False

                models_consistent = models_consistent and predicts_correct

            if models_consistent:
                found_attributes.append(attr)

        remove_attributes = []

        for attr in found_attributes:
            ctr = 0
            for model in model_attr_map.keys():
                if model in reference:
                    continue
                if attr not in model_attr_map[model]:
                    continue
                if (
                    len(model_attr_map[model][attr]) > 0
                    and max(model_attr_map[model][attr]) == 1
                ):
                    ctr += 1
            if ctr > tolerance:
                remove_attributes.append(attr)

        for attr in remove_attributes:
            found_attributes.remove(attr)

        if len(found_attributes) == 0:
            continue
        else:
            filtered_profiles.append(profile)
            print("=" * 80)
            profile.print()

    with SafeOpen(outpath, "w") as f:
        for profile in filtered_profiles:
            profile.to_file(f)


def no_correct(
    path: str, outpath: str, reference: List[str], pii_types: List[str]
) -> None:
    """Checks if the profile has no correct attributes.

    Args:
        profile (Profile): The profile to check
        reference (List[str]): List of model names that should be used as reference
        pii_types (List[str]): List of PII types that should be looked at

    Returns:
        bool: Whether the profile has no correct attributes
    """

    profiles = load_data(path)

    selected_profiles: List[Profile] = []

    for profile in profiles:
        model_attr_map = get_model_attr_map(profile)

        if len(reference) == 0:
            reference = list(model_attr_map.keys())

        found_attributes = []

        all_reviewers = list(profile.review_pii.keys())

        possible_pii = [
            pii
            for pii in pii_types
            if profile.review_pii[all_reviewers[0]][pii]["hardness"] > 0
        ]

        for attr in possible_pii:
            model_can_predict = False
            for model in reference:
                if model not in model_attr_map:
                    break
                if attr not in model_attr_map[model]:
                    break

                if len(model_attr_map[model][attr]) > 0:
                    predicts_correct = max(model_attr_map[model][attr]) == 1
                else:
                    predicts_correct = False

                model_can_predict = model_can_predict or predicts_correct

            if not model_can_predict:
                found_attributes.append(attr)

        if len(found_attributes) > 0:
            selected_profiles.append(profile)
            print("=" * 80)
            print(f"Attributes: {found_attributes}")
            profile.print()

    with SafeOpen(outpath, "w") as f:
        for profile in selected_profiles:
            profile.to_file(f)


def subreddit_required(
    path: str, outpath: str, reference: List[str], pii_types: List[str]
) -> None:
    """All profiles for which reviewers required subreddits

    Args:
        profile (Profile): The profile to check
        reference (List[str]): List of model names that should be used as reference
        pii_types (List[str]): List of PII types that should be looked at

    Returns:
        bool: Whether we required to subreddit
    """

    profiles = load_data(path)

    selected_profiles: List[Profile] = []

    for profile in profiles:
        model_attr_map = get_model_attr_map(profile)

        if len(reference) == 0:
            reference = list(model_attr_map.keys())

        found_attributes = []

        all_reviewers = list(profile.review_pii.keys())

        possible_pii = [
            pii
            for pii in pii_types
            if profile.review_pii[all_reviewers[0]][pii]["hardness"] > 0
        ]

        for attr in possible_pii:
            model_can_predict = False

            for model in reference:
                if model not in model_attr_map:
                    break
                if attr not in model_attr_map[model]:
                    break

                required_subreddit = False
                for reviewer in profile.review_pii.keys():
                    required_subreddit = required_subreddit or (
                        "detect_from_subreddit" in profile.review_pii[reviewer][attr]
                        and profile.review_pii[reviewer][attr]["detect_from_subreddit"]
                    )

                if len(model_attr_map[model][attr]) > 0:
                    predicts_correct = max(model_attr_map[model][attr]) == 1
                else:
                    predicts_correct = False

                model_can_predict = model_can_predict or predicts_correct

            if required_subreddit:
                found_attributes.append(attr)

        if len(found_attributes) > 0:
            selected_profiles.append(profile)
            print("=" * 80)
            print(f"Attributes: {found_attributes}")
            profile.print()

    with SafeOpen(outpath, "w") as f:
        for profile in selected_profiles:
            profile.to_file(f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str)
    args.add_argument("--outpath", type=str)
    args.add_argument("--reference", type=str, nargs="*")
    args.add_argument(
        "--pii_types",
        type=str,
        nargs="+",
        default=[
            "age",
            "education",
            "location",
            "income",
            "pobp",
            "married",
            "gender",
            "occupation",
        ],
    )
    args.add_argument("--no_human", action="store_true")
    args.add_argument("--subreddit_required", action="store_true")
    args.add_argument("--min_hardness", type=int, default=0)
    args.add_argument("--min_certainty", type=int, default=0)
    args.add_argument("--tolerance", type=int, default=0)

    args = args.parse_args()

    if args.no_human:
        no_correct(args.path, args.outpath, args.reference, args.pii_types)
    elif args.subreddit_required:
        subreddit_required(args.path, args.outpath, args.reference, args.pii_types)
    else:
        filter_ds(
            args.path,
            args.outpath,
            args.reference,
            args.pii_types,
            args.no_human,
            args.min_hardness,
            args.min_certainty,
            args.tolerance,
        )
