import argparse
from pandas import DataFrame
from utils.data import load_data
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.reddit.eval import compare_ages, get_model_answers
from src.models.model_factory import get_model
from src.utils.initialization import set_credentials
from src.configs import REDDITConfig, ModelConfig, Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/reddit/results/out_test.jsonl"
    )
    args, unknown = parser.parse_known_args()

    model_config = ModelConfig(
        name="gpt-4",
        provider="openai",
        max_workers=8,
        args={
            "temperature": 0.1,
        },
    )

    reddit_config = None

    config = Config(
        gen_model=model_config,
        task_config=reddit_config,
        store=True,
    )
    set_credentials(config)

    model = get_model(config.gen_model)

    data = load_data(args.path, args)

    indiv_comments = []

    users = {}

    rev_1_name = "<set reviewer name 1>"  # TODO set reviewer names
    rev_2_name = "<set reviewer name 2>"

    # Parse into dataframe with index, reviewer, attribute, hardness, certainty

    df_list = []

    for dp in data:
        for reviewer, review in dp["reviews"].items():
            for attribute, rev_vals in review.items():
                if attribute == "time" or attribute == "timestamp":
                    continue

                df_list.append(
                    {
                        "username": dp["username"],
                        "reviewer": reviewer,
                        "attribute": attribute,
                        "comments": dp["comments"],
                        "model_estimate": (
                            dp["predictions"]["gpt-4"][attribute]["guess"]
                            if attribute in dp["predictions"]["gpt-4"]
                            else ""
                        ),
                        "model_eval": (
                            dp["evaluations"]["gpt-4"][rev_1_name][attribute]
                            if attribute in dp["predictions"]["gpt-4"]
                            else ""
                        ),
                        "estimate": rev_vals["estimate"],
                        "hardness": rev_vals["hardness"],
                        "certainty": rev_vals["certainty"],
                    }
                )

    df = DataFrame(df_list)

    compare_ages("24-28", "24-26")
    # Filter out all username attribute pairs where no reviewer has hardness > 0

    df = df.set_index(["username", "attribute"])
    # Filter out to only rows where certainty is > 2
    df = df.groupby(level=[0, 1]).filter(lambda x: (x["certainty"] > 2).any())

    # Count how many have zero by one reviewer but not the other
    count = 0
    with open("hardness_zeros.txt", "w") as f:
        for name, group in df.groupby(level=[0, 1]):
            # check if one is 0 and the other is not
            if 0 in group["hardness"].values and group["hardness"].nunique() > 1:
                count += 1
                f.write(f"{name}\n")
                f.write(f"{str(group['estimate'].tolist())}\n")
                f.write(f"{str(group['hardness'].tolist())}\n")

                comment_list = group["comments"].tolist()[0]
                comment_list = [(c["subreddit"], c["text"]) for c in comment_list]
                for c in comment_list:
                    f.write(f"{c[0]}: {c[1]}")
                    f.write("\n")
                f.write("\n\n====================\n\n")

    print(f"Number of zeros: {count} out of {len(df.groupby(level=[0,1]))}")

    df = df.groupby(level=[0, 1]).filter(lambda x: (x["hardness"] != 0).all())

    # Find all where one has hardness >=4 and the other has hardness <= 2
    count = 0
    with open("hardness_disagreements.txt", "w") as f:
        for name, group in df.groupby(level=[0, 1]):
            # check if one is 0 and the other is not
            if group["hardness"].max() >= 4 and group["hardness"].min() <= 2:
                count += 1
                f.write(f"{name}\n")
                f.write(f"{str(group['estimate'].tolist())}\n")
                f.write(f"{str(group['hardness'].tolist())}\n")

                comment_list = group["comments"].tolist()[0]
                comment_list = [c["text"] for c in comment_list]
                for c in comment_list:
                    f.write(c)
                    f.write("\n")
                f.write("\n\n====================\n\n")

    # Count how many have a distance greater than 1 and how many have a distance <= 1
    count = 0
    close_count = 0
    for name, group in df.groupby(level=[0, 1]):
        # check if one is 0 and the other is not
        if group["hardness"].max() - group["hardness"].min() > 1:
            count += 1
        else:
            close_count += 1

    # Group by username and and attribute
    pivot_df = df.pivot_table(
        index=["username", "attribute"], columns="reviewer", values=["hardness"]
    )

    rev_1 = pivot_df[("hardness", rev_1_name)]
    rev_2 = pivot_df[("hardness", rev_2_name)]

    sq_dist = np.mean(np.sqrt((rev_1 - rev_2) ** 2))
    kappa_score = cohen_kappa_score(rev_1, rev_2, weights="quadratic")
    pearson_correlation = np.corrcoef(rev_1, rev_2)[0, 1]

    print("Cohen Kappa Score is ", kappa_score)
    print("Pearson Correlation is ", pearson_correlation)

    cm = confusion_matrix(rev_1, rev_2, labels=[1, 2, 3, 4, 5])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.savefig("confusion_matrix.png")

    df["model_eval"] = df["model_eval"].apply(lambda x: [int(i) for i in x])
    df["model_eval"] = df["model_eval"].apply(lambda x: (x + [0] * 3)[:3])

    # Reviewer estimates and calculate their agreement
    agreements = df.groupby(["username", "attribute"]).filter(
        lambda x: x["estimate"].nunique() == 1
    )
    disagreements = df.groupby(["username", "attribute"]).filter(
        lambda x: x["estimate"].nunique() > 1
    )

    # for each user attribute print the disagreement and the values
    for name, group in disagreements.groupby(["username", "attribute"]):
        print(name)
        print(group["estimate"].unique())

        df_reset = group.reset_index()
        attr = df_reset["attribute"][
            0
        ]  # this will get the attribute of the 1st record.

        if attr == "age":
            is_corr = compare_ages(group["estimate"][0], group["estimate"][1])
        else:
            _, is_corr = next(
                get_model_answers(
                    group["estimate"][0], [group["estimate"][1]], model=model
                )
            )
            if is_corr == "yes":
                is_corr = True
            elif is_corr == "no":
                is_corr = False
            elif is_corr == "less precise":
                is_corr = False
            else:
                is_corr = False

        if is_corr:
            # Append to the agreement dataframe
            agreements = pd.concat([agreements, group])
            print("\nCorrect")

        if not is_corr and attr not in ["age", "education", "married", "income"]:
            # Append to the disagreement dataframe
            print(f"\n{group['estimate'][0]} vs. {group['estimate'][1]}")
            user_input = input("Is this correct? (y/n): ")
            if user_input == "y":
                agreements = pd.concat([agreements, group])

    total = len(agreements[agreements["reviewer"] == rev_1_name])
    total_df = len(df[df["reviewer"] == rev_1_name])

    print(f"Agreements: {total} out of {total_df} ({total/total_df})")

    sums = [0, 0, 0]

    for index, elem in agreements[agreements["reviewer"] == rev_1_name].iterrows():
        sums[0] += elem["model_eval"][0]
        if elem["model_eval"][0] != 1:
            sums[1] += elem["model_eval"][1]
            if elem["model_eval"][1] != 1:
                sums[2] += elem["model_eval"][2]

        if sum(elem["model_eval"]) == 0:
            print(elem["model_estimate"])
            print(elem["estimate"])
            user_input = input("Is this correct? X,X,X: ")
            res = user_input.split(",")
            sums[0] += int(res[0])
            sums[1] += int(res[1])
            sums[2] += int(res[2])

    total = len(agreements[agreements["reviewer"] == rev_1_name])
    sums = [i / total for i in sums]

    print(f"Top 1: {sums[0]}")
    print(f"Top 2: {sums[0] + sums[1]}")
    print(f"Top 3: {sums[0] + sums[1] + sums[2]}")

    print(sums)

    # Human eval: We have 26 non-agreements and 4 less precise out of 332 + 160 = 492
