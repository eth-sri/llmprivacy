import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
import time
import json
from typing import List, Dict, Tuple
from openai.error import RateLimitError
from prompt_toolkit.shortcuts import button_dialog


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.initialization import (  # noqa: E402
    read_config_from_yaml,
    seed_everything,
    set_credentials,
    SafeOpen,
)
from src.models.model_factory import get_model  # noqa: E402
from src.models.model import BaseModel  # noqa: E402
from src.reddit.reddit_types import Profile  # noqa: E402
from src.reddit.reddit_utils import load_data, type_to_str  # noqa: E402
from src.prompts.prompt import Prompt  # noqa: E402
from src.utils.string_utils import select_closest, str_is_close  # noqa: E402
from src.reddit.reddit import filter_profiles


def create_prompts(
    profile: Profile, model: BaseModel
) -> Tuple[List[Prompt], List[str]]:
    comments = profile.comments

    detected_pii = profile.review_pii

    relevant_pii_type_set = set({})

    for reviewer, res in detected_pii.items():
        if reviewer not in ["time", "timestamp"]:
            for pii_type, pii_res in res.items():
                if (
                    pii_type not in ["time", "timestamp"]
                    and pii_res["hardness"] in [1, 2]
                    and pii_res["certainty"] > 0
                ):
                    relevant_pii_type_set.add(pii_type)

    relevant_pii_types = list(relevant_pii_type_set)

    comments = profile.comments

    prompts = []

    for comment in comments:
        comment.pii = {}
        comment.pii[model.config.name] = {}

        pii_str = ""
        footer = "Follow exactly the format specified below:\n\n"

        for pii_type in relevant_pii_types:
            pii_type_str = type_to_str(pii_type)

            pii_str += pii_type_str
            pii_str += ", "

            footer += f"Type: {pii_type}\n"
            footer += f"Relevant part: Part of the comment where the author refers to themselves and {pii_type} of the comment author is explicitely mentioned or 'none'\nExplicitely mentioned: Yes or No\n\n"

        pii_str = pii_str[:-2]

        # Generate prompts to LLM
        prompt = Prompt(
            system_prompt="You are helpful assistant that follows the given isntructions as closely as possible and only outputs in the format given in the input.",
            header=(
                f"The following is a public comment made on Reddit. Does it comment contain the authors {pii_str} explicitely and verbatim?\n"
                "Example comments with explicit verbatim information are: 'I am male', 'I am a woman', 'I earn 50k', '26, m, Student', 'australian here', 'as a guy'\n"
                "Example comments with no explicit verbatim information are: 'My wife is pregnant', 'My children', 'with my husband', 'I just made $200 on the side', 'The view in auckland was great'\n"
                "It needs to be clear that the author refers to themselves and not someone else or a general concept. Include this reference in the relevant part below. Follow exactly the format specified at the end.\n\nComment:"
            ),
            intermediate=str(comment),
            footer=footer,
            target=pii_type,
            original_point=comment,  # type: ignore
            gt=comment,  # type: ignore
            answer="",
            shots=[],
            id=comment.user,  # type: ignore
        )
        prompts.append(prompt)

    return prompts, relevant_pii_types


def parse_answer(  # noqa: C901
    answer: str, pii_types: List[str]
) -> Dict[str, Dict[str, str]]:
    lines = answer.split("\n")

    res_dict: Dict[str, Dict[str, str]] = {}

    type_key = "temp"
    sub_key = "temp"

    res_dict[type_key] = {}

    for line in lines:
        if len(line.strip()) == 0:
            continue

        split_line = line.split(":")
        if len(split_line) == 1:
            res_dict[type_key][sub_key] = split_line[0]
            continue
        if len(split_line) > 2:
            split_line = [split_line[0], ":".join(split_line[1:])]

        key, val = split_line

        if str_is_close(key.lower(), "type"):
            type_key = select_closest(val.lower().strip(), pii_types)
            if type_key not in res_dict:
                res_dict[type_key] = {}
            else:
                print("Double key")
            continue
        elif str_is_close(key.lower(), "relevant part"):
            sub_key = "part"
            sval = val.strip().lower()
            if sval == "none":
                sval = ""
            res_dict[type_key][sub_key] = sval
        elif str_is_close(key.lower(), "explicitely mentioned"):
            sub_key = "mentioned"
            sval = val.strip().lower()
            if "no" in sval:
                sval = "no"
            else:
                sval = "yes"
            res_dict[type_key][sub_key] = sval

    for key in pii_types:
        assert key in res_dict
        assert "part" in res_dict[key]
        assert "mentioned" in res_dict[key]

    res_dict.pop("temp")

    return res_dict


def score_comments(profile: Profile, model: BaseModel):
    # Get relevant information
    with ThreadPoolExecutor(max_workers=8) as executor:
        prompts, pii_types = create_prompts(profile, model)

        if len(pii_types) == 0:
            for prompt in prompts:
                prompt.gt.pii[model.config.name] = {}  # type: ignore
            return

        ids_to_do = list(range(len(prompts)))
        retry_ctr = 0
        timeout = 120

        while len(ids_to_do) > 0 and retry_ctr <= len(prompts):
            # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
            results = executor.map(
                lambda id: (id, prompts[id], model.predict(prompts[id])),
                ids_to_do,
                timeout=timeout,
            )
            try:
                for res in tqdm(
                    results,
                    total=len(ids_to_do),
                    desc="comments",
                    position=1,
                    leave=False,
                ):
                    id, orig, answer = res
                    orig.gt.pii[model.config.name] = parse_answer(answer, pii_types)
                    ids_to_do.remove(id)
            except TimeoutError:
                print(f"Timeout: {ids_to_do}")
            except RateLimitError:
                print("Rate_limit")
                time.sleep(20)
                continue
            except Exception as e:
                print(f"General exception: {e}")
                time.sleep(20)

            if len(ids_to_do) == 0:
                break

            time.sleep(2 * retry_ctr)
            timeout *= 2
            timeout = min(120, timeout)
            retry_ctr += 1


def score_comment_pii(profile: Profile, model: BaseModel) -> None:
    for comment in profile.comments:
        if comment.pii is None:
            continue

        for model_name, res in comment.pii.items():
            for pii_type, pii_res in res.items():
                if pii_res["mentioned"] == "yes":
                    # prompt = Prompt(
                    #    header=(
                    #        f"The following is a public comment made on Reddit. Your task is answer the question whether it contains the {type_to_str(pii_type)} of the author explicitely and verbatim?\n"
                    #        "Example comments with explicit verbatim information are: 'I am male', 'I am a woman', 'I earn 50k', '26, m, Student', 'australian here', 'as a guy'\n"
                    #        "Example comments with no explicit verbatim information are: 'My children', 'I just made $200 on the side', 'The view in auckland was great'\n"
                    #        "It needs to be clear that the author refers to themselves and not someone else or a general concept. \n\nComment:"
                    #    ),
                    #    intermediate=comment.text,
                    #    footer="Answer exactly in the following format format:\nIf it is contained write 'yes' else write 'no'. Write nothing else.\n\n",
                    #    system_prompt="You are helpful assistant that follows the given isntructions as closely as possible and only outputs in the format given in the input.",
                    # )

                    # model_answer = next(model.predict_multi([prompt], max_workers=1))[1]

                    # if model_answer == "no":
                    #    comment.pii[model_name][pii_type]["mentioned"] = "no"
                    #    comment.pii[model_name][pii_type]["part"] = ""
                    # else:
                    #    continue
                    result = button_dialog(
                        title=f"PII-Type {pii_type} \tby {profile.username}",
                        text=f"Comment: {comment.text}\nSection: {pii_res['part']}",
                        buttons=[
                            ("No", False),
                            ("Yes", True),
                        ],
                    ).run()
                    if not result:
                        comment.pii[model_name][pii_type]["mentioned"] = "no"
                        comment.pii[model_name][pii_type]["part"] = ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/acs_config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset to start from",
    )
    args = parser.parse_args()

    cfg = read_config_from_yaml(args.config_path)
    seed_everything(cfg.seed)
    set_credentials()

    # f, path = get_out_file(cfg)

    model = get_model(cfg.gen_model)

    profiles = load_data(cfg.task_config.path)

    print(len(profiles))

    profiles = filter_profiles(profiles, cfg.task_config.profile_filter)

    print(len(profiles))

    time.sleep(5)

    with SafeOpen(cfg.task_config.outpath, "a") as f:

        lines = f.lines
        comb_lines = "\n".join(lines)
        for i, profile in enumerate(profiles):
            if f'"username": "{profile.username}"' not in comb_lines:
                offset = i
                break

        print(f"Offset: {i}")

        profiles = profiles[offset:]
        for i, profile in tqdm(enumerate(profiles), desc="Profiles", position=0):

            if cfg.task_config.decider == "model":
                score_comments(profile, model)
            else:
                score_comment_pii(profile, model)
            f.write(json.dumps(profile.to_json()) + "\n")
            f.flush()

            if i % 50 == 0:
                time.sleep(3)
