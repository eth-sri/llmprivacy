import argparse
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Optional
import sys
import os

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.visualization.visualize_reddit import profiles_to_df, filter_and_format_df
from src.utils.initialization import SafeOpen
from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Comment, Profile
from cmcrameri import cm

color_map = cm.managua  # sns.color_palette("muted", len(sorted_attributes))
spaced_points = np.linspace(0, 1, 10)
colors = [color_map(i) for i in reversed(spaced_points[:8])]

# sns.set_palette(palette=color_map)

map_model_name = {
    "gpt-4": "GPT-4",
    "gpt-3.5-turbo-16k-0613": "GPT-3.5",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7b",
    "meta-llama/Llama-2-13b-chat-hf": "Llama-2-13b",
    "meta-llama/Llama-2-70b-chat-hf": "Llama-2-70b",
    "chat-bison@001": "PaLM 2 Chat",
    "text-bison@001": "PaLM 2 Text",
    "Claude-2-100k": "Claude-2",
    "Claude-instant-100k": "Claude-Instant",
}

model_order = [
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "PaLM 2 Text",
    "PaLM 2 Chat",
    "Claude-2",
    "Claude-Instant",
    "GPT-3.5",
    "GPT-4",
]

sorted_attributes = [
    "gender",
    "location",
    "married",
    "age",
    "education",
    "occupation",
    "pobp",
    "income",
]


def write_histplot(df, x, bins, discrete, kde, kde_kws, path):
    sns.histplot(
        data=df,
        x=x,
        bins=bins,
        discrete=discrete,
        kde=kde,
        kde_kws=kde_kws,
        color=colors[0],
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def write_table(
    df: DataFrame,
    rows,
    columns,
    values="value",
    agg="count",
    type="float",
    path="plots/dataset/stat_temp.txt",
):
    if type == "float":
        formatter = "{:.2f}".format
    elif type == "int":
        formatter = lambda x: int(x)

    complete_pivot = df.pivot_table(
        index=rows, columns=columns, values=values, aggfunc=agg
    )

    with open(path, "w") as f:
        f.write(
            complete_pivot.to_latex(
                formatters={"name": str.upper}, na_rep="-", float_format=formatter
            )
        )


def plot_dcont(df, x, x_legend, hue, path):
    plt.figure(figsize=(10, 6))
    g = sns.histplot(
        data=df,
        x=x,
        hue=hue,
        bins=100,
        color=[colors[0], colors[2], colors[4], colors[6]],
    )
    plt.title("")

    plt.tick_params(axis="x", labelsize=16)  # change the size of x ticks
    plt.tick_params(axis="y", labelsize=16)  # change the size of y ticks

    plt.xlabel(x_legend, fontsize=20, fontweight="bold")
    plt.ylabel("Count", fontsize=20, fontweight="bold")

    # Get the legend
    legend = g.get_legend()
    legend.set_title("")
    # Set the fontsize
    for text in legend.texts:
        text.set_fontsize("20")

    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def get_decont_stats(path: str):
    models = []
    similarities = []
    token_equality = []
    suf_pref_match = []
    long_substr = []
    bleu = []

    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r") as file:
                lines = file.read().splitlines()

                model_name = "_".join(filename.split(".")[0].split("_")[1:])
                models.append(model_name)

                similarities.append(eval(lines[-5]))
                token_equality.append(eval(lines[-4]))
                suf_pref_match.append(eval(lines[-3]))
                long_substr.append(eval(lines[-2]))
                bleu.append(eval(lines[-1]))

    num_models = len(models)

    print(models)

    data = []

    for i in range(len(token_equality[0])):
        for j in range(0, num_models):
            point = {"index": i}
            point["model"] = models[j]
            point["sim"] = similarities[j][i]
            point["token_eq"] = token_equality[j][i]
            point["sufpref"] = suf_pref_match[j][i]
            point["substr"] = long_substr[j][i]

            bleu_val = float(bleu[j][i])
            if float(bleu[j][i]) < 0.01:
                bleu_val = 0.0

            point["bleu"] = bleu_val

            data.append(point)

    df = pd.DataFrame(data)

    print(df)

    # df = df[df["model"] != "chat_bison"]

    plot_dcont(df, "sim", "String Similarity Ratio", "model", "plots/decont_sim.png")
    plot_dcont(df, "token_eq", "Number of Equal Tokens", "model", "plots/decont_eq.png")
    plot_dcont(
        df, "sufpref", "Longest Prefix Match", "model", "plots/decont_sufpref.png"
    )
    plot_dcont(
        df, "substr", "Longest Substring Match", "model", "plots/decont_substr.png"
    )
    plot_dcont(df, "bleu", "BLEU Score", "model", "plots/decont_bleu.png")


def get_dataset_stats(path: str):
    profiles = load_data(path)
    df = profiles_to_df(profiles)

    first_index_df = df[df["index"] == 0]

    # Get the first model
    first_model = first_index_df.groupby(["model"]).sum().reset_index()

    first_index_df = first_index_df[first_index_df["model"] == first_model["model"][0]]

    sns.set_theme(style="darkgrid")
    sns.set_palette("pastel")

    target_attr = "all"

    if target_attr != "all":
        first_index_df = first_index_df[first_index_df["attribute"] == target_attr]

    bins = [1, 2, 3, 4, 5]
    kde_kws = {"bw_adjust": 2}
    # Hardness and certainty dist

    # Overall
    write_histplot(
        first_index_df,
        "hardness",
        bins,
        True,
        True,
        kde_kws,
        "plots/dataset/hist_hardness.png",
    )
    write_histplot(
        first_index_df,
        "certainty",
        bins,
        True,
        True,
        kde_kws,
        "plots/dataset/hist_certainty.png",
    )

    # Per attribute
    for attr in sorted_attributes:
        filtered_df = first_index_df[first_index_df["attribute"] == attr]

        write_histplot(
            filtered_df,
            "hardness",
            bins,
            True,
            True,
            kde_kws,
            f"plots/dataset/hist_hardness_{attr}.png",
        )
        write_histplot(
            filtered_df,
            "certainty",
            bins,
            True,
            True,
            kde_kws,
            f"plots/dataset/hist_certainty_{attr}.png",
        )

    write_table(
        first_index_df,
        rows=["hardness", "certainty"],
        columns=["attribute"],
        type="int",
        path="plots/dataset/hard_cert_table.txt",
    )
    write_table(
        first_index_df,
        rows=["hardness"],
        columns=["attribute"],
        type="int",
        path="plots/dataset/hard_table.txt",
    )
    write_table(
        first_index_df,
        rows=["certainty"],
        columns=["attribute"],
        type="int",
        path="plots/dataset/cert_table.txt",
    )

    # Comment distribution
    comment_data = []

    for pr in profiles:
        for comment in pr.comments:
            row = {
                "username": pr.username,
                "comment": comment.text,
                "subreddit": comment.subreddit,
            }
            comment_data.append(row)

    comment_df = pd.DataFrame(comment_data)
    comment_df["comment_length"] = comment_df["comment"].apply(len)

    comment_counts = comment_df.groupby("username").count().reset_index()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=comment_counts, x="comment", bins=30, color=colors[0])
    plt.title("Histogram of Number of Comments Per User")
    plt.xlabel("Number of Comments")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig("plots/dataset/comments.png")

    plt.clf()
    print("Hello")

    # Comment_length distribution
    length_counts = comment_df.groupby("username")["comment_length"].sum().reset_index()

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=length_counts, x="comment_length", bins=30, color=colors[0])
    plt.title("Histogram of Total Comment Lengths Per User")
    plt.xlabel("Total Length of Comments by User (#Characters)")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig("plots/dataset/comments_length.png")
    plt.clf()

    # Subreddit distribution
    counts = comment_df["subreddit"].value_counts().nlargest(50)

    # Create a DataFrame suitable for Seaborn
    counts_df = pd.DataFrame(counts).reset_index()
    counts_df.columns = ["subreddit", "number_of_comments"]

    plt.figure(figsize=(9, 9))
    sns.barplot(
        y="subreddit",
        x="number_of_comments",
        data=counts_df,
        color=colors[0],
    )
    plt.title("Number of comments in each subreddit (Top 50)")
    plt.xlabel("Number of comments")
    plt.ylabel("Subreddit")
    plt.tight_layout()
    plt.savefig("plots/dataset/comments_subreddit.png")
    plt.clf()


def get_specfic_stats(
    path: str,
    min_hardness: int = 0,
    min_certainty: int = 0,
    max_hardness: Optional[int] = None,
    max_certainty: Optional[int] = None,
    top_k: int = 1,
    target_attr: str = "all",
    models: List[str] = [],
):
    profiles = load_data(path)
    df = profiles_to_df(profiles)

    df = filter_and_format_df(
        df,
        min_hardness,
        min_certainty,
        top_k,
        target_attr,
        models,
        max_hardness,
        max_certainty,
    )

    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

    # Now you can sort by 'Grade' in the specified order
    df = df.sort_values("model")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    complete = df.groupby(["model", "attribute", "hardness", "certainty"])
    # complete = complete["sum_value"].describe()

    complete = complete["sum_value"].mean()
    complete_pivot = complete.to_frame().pivot_table(
        index=["model", "hardness", "certainty"], columns="attribute"
    )

    print("Model - Attribute - Hardness - Certainty".center(80, "-"))
    print(
        complete_pivot.to_latex(
            formatters={"name": str.upper}, na_rep="-", float_format="{:.2f}".format
        )
    )

    print("\n\n\n\n\n" + "Model - Attribute - Hardness".center(80, "-"))
    hardness = df.groupby(["model", "attribute", "hardness"])["sum_value"].mean()
    hardness_pivot = hardness.to_frame().pivot_table(
        index=["model", "hardness"], columns="attribute"
    )
    print(
        hardness_pivot.to_latex(
            formatters={"name": str.upper}, na_rep="-", float_format="{:.2f}".format
        )
    )

    print("\n\n\n\n\n" + "Model - Attribute - Certainty".center(80, "-"))
    certainty = df.groupby(["model", "attribute", "certainty"])["sum_value"].mean()
    certainty_pivot = certainty.to_frame().pivot_table(
        index=["model", "certainty"], columns="attribute"
    )
    print(
        certainty_pivot.to_latex(
            formatters={"name": str.upper}, na_rep="-", float_format="{:.2f}".format
        )
    )

    print("\n\n\n\n\n" + "Model - Attribute".center(80, "-"))
    model = df.groupby(["model", "attribute"])["sum_value"].mean()
    model_pivot = model.to_frame().pivot_table(index=["model"], columns="attribute")
    print(
        model_pivot.to_latex(
            formatters={"name": str.upper}, na_rep="-", float_format="{:.2f}".format
        )
    )

    # NExt
    print("\n\n\n\n\n" + "Model - Hardness".center(80, "-"))
    hardness = df.groupby(["model", "attribute", "hardness"])["sum_value"].mean()
    hardness_pivot = hardness.to_frame().pivot_table(
        index=["hardness"], columns=["model"]
    )
    print(
        hardness_pivot.to_latex(
            index=False,
            formatters={"name": str.upper},
            na_rep="-",
            float_format="{:.2f}".format,
        )
    )

    print("\n\n\n\n\n" + "Model - Certainty".center(80, "-"))
    certainty = df.groupby(["model", "attribute", "certainty"])["sum_value"].mean()
    certainty_pivot = certainty.to_frame().pivot_table(
        index=["certainty"], columns=["model"]
    )
    print(
        certainty_pivot.to_latex(
            formatters={"name": str.upper}, na_rep="-", float_format="{:.2f}".format
        )
    )

    print("\n\n\n\n\n" + "Model".center(80, "-"))
    model = df.groupby(["model", "attribute"])["sum_value"].mean()
    model_pivot = model.to_frame().pivot_table(index=[], columns=["model"])
    print(
        model_pivot.to_latex(
            formatters={"name": str.upper}, na_rep="-", float_format="{:.2f}".format
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        default=["eval_results/full_eval_model_human.jsonl"],
    )
    parser.add_argument("--models", type=str, nargs="*", default=[])
    parser.add_argument("--attributes", type=str, nargs="*", default=[])
    parser.add_argument("--min_hardness", type=int, default=0)
    parser.add_argument("--min_certainty", type=int, default=0)
    parser.add_argument("--max_hardness", type=int, default=None)
    parser.add_argument("--max_certainty", type=int, default=None)
    parser.add_argument("--plot_dataset", action="store_true")
    parser.add_argument("--plot_decontamination", action="store_true")
    parser.add_argument("--plot_specific", action="store_true")
    args, unknown = parser.parse_known_args()

    if len(args.path) == 1:
        args.path = args.path[0]

    if args.plot_dataset:
        get_dataset_stats(args.path)
    if args.plot_decontamination:
        get_decont_stats(args.path)
    if args.plot_specific:
        get_specfic_stats(
            args.path,
            min_hardness=args.min_hardness,
            min_certainty=args.min_certainty,
            max_hardness=args.max_hardness,
            max_certainty=args.max_certainty,
            top_k=1,
            target_attr="all",
            models=args.models,
        )
