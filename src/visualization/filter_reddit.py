import argparse
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Dict, Tuple, List
import sys

from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.initialization import SafeOpen
from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Comment, Profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="eval_results/full_eval_model_human.jsonl"
    )
    parser.add_argument("--model", type=str, default="gpt-4")
    args, unknown = parser.parse_known_args()

    profiles = load_data(args.path)

    for i, profile in enumerate(profiles):
        print(f"====== Profile {i} ======")
        print(profile.print_model_response(args.model))
