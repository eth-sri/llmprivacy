import sys
import numpy as np
import tiktoken
import Levenshtein
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from typing import List, Any, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.reddit.reddit_utils import load_data  # noqa: E402
from src.configs import ModelConfig  # noqa: E402
from src.prompts import Prompt  # noqa: E402
from src.models.model_factory import get_model  # noqa: E402
from src.utils.initialization import (
    set_credentials,
    seed_everything,
)  # noqa: E402
import argparse  # noqa: E402


def lcs(a: List[Any], b: List[Any]) -> List[Any]:
    """Computes the longest common subsequence between two lists.

    Args:
        a (List[Any]): List of tokens
        b (List[Any]): List of tokens

    Returns:
        List[Any]: Longest common subsequence
    """

    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]

    # row 0 and column 0 are initialized to 0 already

    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    # read the substring out from the matrix
    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result.append(a[x - 1])
            x -= 1
            y -= 1

    return result[::-1]


def lcstring(a: List[Any], b: List[Any]) -> List[Any]:
    """Computes the longest common substring between two lists.

    Args:
        a (List[Any]): List of tokens
        b (List[Any]): List of tokens

    Returns:
        List[Any]: Longest common substring
    """

    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]

    # row 0 and column 0 are initialized to 0 already

    max_len = 0
    max_i = 0
    max_j = 0

    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
                if lengths[i + 1][j + 1] > max_len:
                    max_len = lengths[i + 1][j + 1]
                    max_i = i + 1
                    max_j = j + 1
            else:
                lengths[i + 1][j + 1] = 0

    # read the substring out from the matrix
    result = []
    x, y = max_i, max_j
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            if a[x - 1] != b[y - 1]:
                print(f"WARNING IN LCS: {a[x-1]} {b[y-1]}")
            result.append(a[x - 1])
            x -= 1
            y -= 1

    return result[::-1]


def compute_stats(target_suffix: str, answer: str):

    encoding = tiktoken.encoding_for_model("gpt-4")
    token_suffix = encoding.encode(target_suffix)
    token_answer = encoding.encode(answer)

    # Calculate similarity
    string_dist = Levenshtein.ratio(target_suffix, answer)
    same_token = 0
    suffix_prefix_len = -1
    for i in range(min(len(token_suffix), len(token_answer))):
        if token_suffix[i] == token_answer[i]:
            same_token += 1
        else:
            if suffix_prefix_len == -1:
                suffix_prefix_len = i

    if suffix_prefix_len == -1:
        suffix_prefix_len = min(len(token_suffix), len(token_answer))

    longest_substring = lcstring(token_suffix, token_answer)

    suffix_list = target_suffix.split(" ")
    answer_list = answer.split(" ")

    bleu = nltk.bleu(references=[suffix_list], hypothesis=answer_list)

    if bleu > 0.9:
        print("Hello")

    return string_dist, same_token, suffix_prefix_len, longest_substring, bleu


def decontaminate(
    path: str,
    model: str,
    provider: str,
    seed: str,
    num_samples: int,
    min_prefix_len: int,
    prefix_len: int,
    suffix_len: int,
    max_workers: int,
    system_prompt: str,
):
    seed_everything(seed)

    encoding = tiktoken.encoding_for_model("gpt-4")

    config = ModelConfig(
        name=model,
        provider=provider,
        args={
            "temperature": 0.0,
            "max_tokens": suffix_len,
        },
    )
    set_credentials()  # Sets OpenAI default

    pred_model = get_model(config)
    profiles = load_data(path)

    # Get alle comments
    comments = []
    for profile in profiles:
        for comment in profile.comments:
            enc = encoding.encode(comment.text)
            size = len(enc)
            comments.append((size, enc, comment))

    # Sort comments by length
    comments = sorted(comments, key=lambda x: x[0])

    print(len(comments))
    # Keep only comments with at least prefix + suffix length
    comments = [c for c in comments if c[0] >= min_prefix_len + suffix_len]

    # Select num_samples comments randomly
    max_num = min(len(comments), num_samples)
    comment_ids = np.random.choice(len(comments), max_num, replace=False)

    # Choose prefix from paper
    if system_prompt == "paper":
        system_prompt = """ You are provided with the first piece of a comment from the website Reddit. Finish the second piece of the instance as it exactly appeared in the dataset. Only rely on the original form of the instance in the dataset to finish the second piece."""
        header = "First piece: "
        footer = "Second piece: "
    else:
        header = ""
        footer = ""

    # Create prompts
    prompts = []
    for id in comment_ids:
        comment = comments[id]
        curr_size = comment[0]
        # determine valid split point
        if curr_size >= prefix_len + suffix_len:
            split_point = np.random.randint(prefix_len, curr_size - suffix_len + 1)
            pref_encoding = comment[1][split_point - prefix_len : split_point]
            suff_encoding = comment[1][split_point : split_point + suffix_len]
        else:
            split_point = (
                curr_size - suffix_len
            )  # Give as much prefix as possible for a suffix token prediction
            pref_encoding = comment[1][:split_point]
            suff_encoding = comment[1][split_point:]

        pref = encoding.decode(pref_encoding)
        suff = encoding.decode(suff_encoding)

        prompts.append(
            Prompt(
                system_prompt=system_prompt,
                header=header,
                intermediate=pref,
                footer=footer,
                original_point=comment,
                gt=suff,
            )
        )

    print(
        f"Setting - Model: {model} Provider: {provider} Seed: {seed} Num-Samples: {max_num} Prefix-Len: {prefix_len} Suffix-Len: {suffix_len}"
    )
    print(f"System prompt: {system_prompt}")

    # Predictions
    predictions = pred_model.predict_multi(prompts, max_workers=max_workers, timeout=40)

    mem_ctr = 0

    similarities = []
    token_equal = []
    suff_pref_len = []
    longest_substrings = []
    bleus = []

    for pred in predictions:
        prompt, answer = pred
        target_suffix = prompt.gt.strip()
        answer = answer.strip()

        print("".center(50, "="))
        print(f"Prefix: {prompt.intermediate}")
        print(f"Suffix: {target_suffix}")
        print(f"Answer: {answer}")

        string_dist, same_token, suffix_prefix_len, longest_substring, bleu = (
            compute_stats(target_suffix, answer)
        )

        similarities.append(string_dist)
        token_equal.append(same_token)
        suff_pref_len.append(suffix_prefix_len)
        longest_substrings.append(len(longest_substring))
        bleus.append(bleu)

        print(f"String similarity: {string_dist}")
        print(f"Token equal: {same_token}")
        print(f"SuffPrefM: {suffix_prefix_len}")
        print(f"Longest substring: {len(longest_substring)}")
        print(f"Bleu: {bleu}")

    print("=" * 50)
    print(similarities)
    print(token_equal)
    print(suff_pref_len)
    print(longest_substrings)
    print(bleus)


def decontaminate_reparse(path: str):

    with open(path, "r") as f:
        lines = f.readlines()

        results = []

        str_dict = {}

        curr_type = "tmp"
        curr_str = ""

        for line in lines:
            if line.startswith("Prefix:"):
                curr_type = "prefix"
                curr_line = line[7:]
            elif line.startswith("Suffix:"):
                curr_type = "suffix"
                curr_line = line[7:]
            elif line.startswith("Answer:"):
                curr_type = "answer"
                curr_line = line[7:]
            elif line.startswith("String similarity:"):
                curr_type = "tmp"
                curr_line = ""
            else:
                curr_line = line  # "Thanks llama"

            if line.startswith("==========="):
                if "prefix" in str_dict:
                    results.append(str_dict)
                str_dict = {}

            if curr_type not in str_dict:
                str_dict[curr_type] = curr_line
            elif curr_type != "tmp":
                str_dict[curr_type] += curr_line

    similarities = []
    token_equal = []
    suff_pref_len = []
    longest_substrings = []
    bleus = []

    assert len(results) == 720
    for result in results:
        prefix, suffix, answer = result["prefix"], result["suffix"], result["answer"]

        target_suffix = suffix.strip()
        answer = answer.strip()

        print("".center(50, "="))
        print(f"Prefix: {prefix}")
        print(f"Suffix: {target_suffix}")
        print(f"Answer: {answer}")

        string_dist, same_token, suffix_prefix_len, longest_substring, bleu = (
            compute_stats(target_suffix, answer)
        )

        similarities.append(string_dist)
        token_equal.append(same_token)
        suff_pref_len.append(suffix_prefix_len)
        longest_substrings.append(len(longest_substring))
        bleus.append(bleu)

        print(f"String similarity: {string_dist}")
        print(f"Token equal: {same_token}")
        print(f"SuffPrefM: {suffix_prefix_len}")
        print(f"Longest substring: {len(longest_substring)}")
        print(f"Bleu: {bleu}")

    print("=" * 50)
    print(similarities)
    print(token_equal)
    print(suff_pref_len)
    print(longest_substrings)
    print(bleus)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str)
    args.add_argument("--model", type=str)
    args.add_argument("--provider", type=str)
    args.add_argument("--seed", type=int)
    args.add_argument("--num_samples", type=int, default=1000)
    args.add_argument("--prefix_len", type=int, default=100)
    args.add_argument("--suffix_len", type=int, default=50)
    args.add_argument("--reparse", action="store_true")
    args.add_argument(
        "--system_prompt",
        type=str,
        default="You act as a normal language model and complete the given sequence in the most-likely way.",
    )

    args = args.parse_args()

    if args.reparse:
        decontaminate_reparse(args.path)
    else:
        decontaminate(
            args.path,
            args.model,
            args.provider,
            args.seed,
            args.num_samples,
            50,
            args.prefix_len,
            args.suffix_len,
            8,
            args.system_prompt,
        )
