import argparse
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Optional
import sys
import os

import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.initialization import SafeOpen
from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Comment, Profile
from cmcrameri import cm


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

color_map = cm.managua  # sns.color_palette("muted", len(sorted_attributes))
spaced_points = np.linspace(0, 1, 10)
colors = [color_map(i) for i in reversed(spaced_points[:8])]


def store(
    folder,
    prefix,
    models,
    top_k,
    target_attr,
    min_hardness,
    min_certainty,
    max_hardness,
    max_certainty,
    df,
):
    model_df = df.groupby(["model"])
    attribute_df = df.groupby(["model", "attribute"])
    hardness_df = df.groupby(["model", "hardness"])
    certainty_df = df.groupby(["model", "certainty"])

    dfs = [model_df, attribute_df, hardness_df, certainty_df]

    path = f"plots/reddit/{folder}/{prefix}"

    if len(models) > 0:
        model_str = ""
        for model in models:
            model_str += f"{model.split('/')[-1]}_"
        path = f"{path}/{model_str}"
    else:
        path = f"{path}/all"

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(
        f"{path}/full_att={target_attr}_k={top_k}_mih={min_hardness}_mah={max_hardness}_mic={min_certainty}_mac={max_certainty}.png",
        dpi=300,
    )
    plt.close()

    with SafeOpen(f"{path}/stats.txt") as f:
        f.write(
            f"full_att={target_attr}_k={top_k}_mih={min_hardness}_mah={max_hardness}_mic={min_certainty}_mac={max_certainty}\n"
        )

        for stat_df in dfs:
            f.write(stat_df["sum_value"].describe().to_string())
            f.write("\n")
            f.write(stat_df["total"].describe().to_string())
            f.write("\n")


def profiles_to_df(profiles: List[Profile]) -> DataFrame:
    df = DataFrame()
    data = []
    for profile in profiles:
        user = profile.username
        for model, reviewers in profile.evaluations.items():
            for reviewer, attributes in reviewers.items():
                for attribute, values in attributes.items():
                    hardness = profile.review_pii[reviewer][attribute]["hardness"]
                    certainty = profile.review_pii[reviewer][attribute]["certainty"]
                    for i in range(3):
                        if i >= len(values):
                            value = 0
                        else:
                            value = values[i]

                        model_name = map_model_name[model]

                        row = {
                            "username": user,
                            "model": model_name,
                            "reviewer": reviewer,
                            "attribute": attribute,
                            "hardness": hardness,
                            "certainty": certainty,
                            "value": value,
                            "index": i,
                        }
                        data.append(row)

    df = pd.DataFrame(data)
    return df


def filter_and_format_df(
    df: DataFrame,
    min_hardness: int,
    min_certainty: int,
    top_k: int,
    target_attr: str | List[str],
    models: List[str],
    max_hardness: Optional[int] = None,
    max_certainty: Optional[int] = None,
) -> DataFrame:
    # Only ones with hardness and certainty above threshold
    if max_hardness is not None:
        df = df[df["hardness"] <= max_hardness]
    if max_certainty is not None:
        df = df[df["certainty"] <= max_certainty]
    df = df[(df["hardness"] >= min_hardness) & (df["certainty"] >= min_certainty)]
    df = df.astype({"value": "float64"})
    df = df[df["index"] < top_k]
    # Only keep the 1 entry for values clearing 0.5

    precise_df = df.copy()
    precise_df.loc[precise_df["value"] == 0.5, "value"] = 0

    less_precise_df = df.copy()
    less_precise_df.loc[less_precise_df["value"] == 1, "value"] = 0
    less_precise_df.loc[less_precise_df["value"] == 0.5, "value"] = 1

    # Top k rows
    filtered_df = precise_df
    filtered_inprecise_df = less_precise_df
    grouped_sum = filtered_df.groupby(
        ["username", "model", "reviewer", "attribute", "hardness", "certainty"]
    )["value"].sum()
    grouped_inprecise_sum = filtered_inprecise_df.groupby(
        ["username", "model", "reviewer", "attribute", "hardness", "certainty"]
    )["value"].sum()

    grouped_sum = np.clip(grouped_sum, a_min=0, a_max=1)
    grouped_inprecise_sum = np.clip(grouped_inprecise_sum, a_min=0, a_max=1)

    # Drop value in df
    df = df.drop(columns=["value"])
    df = df.drop(columns=["index"])

    df = df.merge(
        grouped_sum.rename("sum_value"),
        left_on=["username", "model", "reviewer", "attribute", "hardness", "certainty"],
        right_index=True,
        how="left",
    )
    df = df.merge(
        grouped_inprecise_sum.rename("impr_sum_value"),
        left_on=[
            "username",
            "model",
            "reviewer",
            "attribute",
            "hardness",
            "certainty",
        ],
        right_index=True,
        how="left",
    )
    df["total"] = df["sum_value"] + df["impr_sum_value"]
    df["total"] = np.clip(df["total"], a_min=0, a_max=1)

    # Now contains the sum of the top k values for each attribute
    if isinstance(target_attr, str):
        if target_attr != "all":
            df = df[df["attribute"] == target_attr]
    else:
        assert isinstance(target_attr, list)
        if not "all" in target_attr:
            df = df[df["attribute"].isin(target_attr)]

    # Get the sum of the values for each model
    # first_index_df = first_index_df.groupby(["model"]).sum().reset_index()

    if len(models) > 0:
        df = df[df["model"].isin(models)]

    return df


def get_specfic_stats(
    path: str,
    min_hardness: int = 0,
    min_certainty: int = 0,
    max_hardness: Optional[int] = None,
    max_certainty: Optional[int] = None,
    top_k: int = 1,
    target_attr: str = "all",
    folder: str = "full",
    models: List[str] = [],
    show_less_precise: bool = False,
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

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    complete = df.groupby(["model", "attribute", "hardness", "certainty"])
    # complete = complete["sum_value"].describe()

    print("Complete".center(80, "-"))
    print(complete["sum_value"].describe())

    hardness = df.groupby(["model", "attribute", "hardness"])
    hardness = hardness["sum_value"].describe()

    print("Hardness".center(80, "-"))
    print(hardness)

    certainty = df.groupby(["model", "attribute", "certainty"])
    certainty = certainty["sum_value"].describe()

    print("Certainty".center(80, "-"))
    print(certainty)


def plot_dataset(path: str, target_attr: str = "all"):
    profiles = load_data(path)
    df = profiles_to_df(profiles)

    first_index_df = df[df["index"] == 0]

    # Get the first model
    fist_model = first_index_df.groupby(["model"]).sum().reset_index()

    first_index_df = first_index_df[first_index_df["model"] == fist_model["model"][0]]

    sns.set_theme(style="darkgrid")
    sns.set_palette("pastel")

    if target_attr != "all":
        first_index_df = first_index_df[first_index_df["attribute"] == target_attr]

    # Aggregate by hardness and certainty, summing the values
    # first_index_df = (
    #     first_index_df.groupby(["hardness", "certainty"]).sum().reset_index()
    # )

    # Create scatter plot with
    plt.figure(figsize=(15, 6))

    g = sns.violinplot(
        x="attribute",
        y="hardness",
        hue="certainty",
        data=first_index_df,
        inner="stick",
        cut=0,
        bw=0.2,
    )
    plt.tight_layout()

    plt.savefig("test.png")
    print("Done")


def plot_hardness(
    path: str,
    min_hardness: int = 0,
    min_certainty: int = 0,
    max_hardness: int = 5,
    max_certainty: int = 5,
    top_k: int = 1,
    target_attr: str = "all",
    folder: str = "full",
    models: List[str] = [],
    show_less_precise: bool = False,
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
        max_hardness=max_hardness,
        max_certainty=max_certainty,
    )

    models_sum = df.groupby("model")["sum_value"].sum()
    sorted_models = models_sum.sort_values(ascending=True).index.tolist()

    # Use hue
    if len(sorted_models) > 4:
        plt.figure(figsize=(15, 7))
    else:
        plt.figure(figsize=(10, 7))
    sns.set_theme(style="darkgrid")
    # sns.set_palette(colors)

    if show_less_precise:
        g = sns.barplot(
            x="model",
            y="total",
            hue="hardness",
            data=df,
            order=sorted_models,
            errorbar=None,
            errcolor="grey",
            errwidth=0.5,
            alpha=0.3,
            palette=colors,
        )
        plt.legend([], [], frameon=False)

    h = sns.barplot(
        x="model",
        y="sum_value",
        hue="hardness",
        data=df,
        order=sorted_models,
        palette=colors,
        errorbar="ci",
        errcolor="grey",
        errwidth=0.5,
    )
    # Rotate x-axis' text
    # plt.xticks(rotation=45)
    # Make title bold and bigger
    plt.title(
        f"",
        fontweight="bold",
        fontsize=14,
    )
    q, r = h.get_legend_handles_labels()
    plt.legend(q[-5:], r[-5:], frameon=True, title="Hardness", ncol=5, loc="upper left")
    plt.xlabel("", fontdict={"weight": "bold", "size": 13})
    plt.ylabel("Accuracy", fontdict={"weight": "bold", "size": 13})
    plt.tight_layout()
    # plt.show()

    # Make x labels bold
    for label in h.get_xticklabels():
        label.set_fontweight("bold")
        label.set_size(13)

    # Make y labels bold
    for label in h.get_yticklabels():
        label.set_fontweight("bold")
        label.set_size(13)

    store(
        folder=folder,
        prefix="hardness",
        models=models,
        top_k=top_k,
        target_attr=target_attr,
        min_hardness=min_hardness,
        min_certainty=min_certainty,
        max_hardness=max_hardness,
        max_certainty=max_certainty,
        df=df,
    )


def plot_stack(
    path: str,
    min_hardness: int = 0,
    min_certainty: int = 0,
    max_hardness: int = 5,
    max_certainty: int = 5,
    top_k: int = 1,
    target_attr: str = "all",
    folder: str = "full",
    models: List[str] = [],
    show_less_precise: bool = False,
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
        max_hardness=max_hardness,
        max_certainty=max_certainty,
    )

    models_sum = df.groupby("model")["sum_value"].sum()
    sorted_models = models_sum.sort_values(ascending=True).index.tolist()

    # Use hue
    if len(sorted_models) > 3:
        plt.figure(figsize=(15, 7))
    else:
        plt.figure(figsize=(10, 7))
    sns.set_theme(style="darkgrid")
    sns.set_palette("pastel")

    # Sort attributes but their total contribution
    attributes_sum = df.groupby("attribute")["total"].sum()
    sorted_attributes = attributes_sum.sort_values(ascending=True).index.tolist()

    removed_df = df.copy()

    # colors = sns.color_palette("flare", len(sorted_attributes))
    # lipari, devon, lapaz, davos

    # Create a fake model which has everything correct
    fake_df = df[df["model"] == sorted_models[0]].copy()
    fake_df["model"] = "Human-Labeled*"
    fake_df["sum_value"] = 1
    fake_df["impr_sum_value"] = 1
    fake_df["total"] = 1

    removed_df = pd.concat([removed_df, fake_df])
    sorted_models = sorted_models + ["Human-Labeled*"]

    max_val = fake_df["sum_value"].sum()

    if top_k > 1:
        max_val = int(max_val / top_k)

    for idx, attribute in enumerate(sorted_attributes):
        # Change color scheme
        g = sns.barplot(
            x="model",
            y="sum_value",
            color=colors[-idx - 1],
            data=removed_df,
            order=sorted_models,
            errorbar=None,
            label=attribute,
        )
        # Remove the current attribute and re-compute the total
        removed_df.loc[removed_df["attribute"] == attribute, "sum_value"] = 0
        plt.legend([], [], frameon=False)

    # g.set_xticks(g.get_xticks()[1:])
    plt.ylim(0, 1)

    target_ticks = [i for i in range(0, max_val, 250)] + [max_val]
    target_ticks_scaled = [t / max_val for t in target_ticks]

    plt.yticks(target_ticks_scaled, target_ticks)
    # Set legend

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles[::-1],
        labels[::-1],
        frameon=True,
        title=None,
        ncol=len(labels),
        loc="center",
        bbox_to_anchor=(0.5, 1.05),
        fontsize=13,
    )
    plt.xlabel("")
    plt.ylabel("# Correct", fontdict={"weight": "bold", "size": 13})
    plt.tight_layout()
    # Storing

    # Make x labels bold
    for label in g.get_xticklabels():
        label.set_fontweight("bold")
        label.set_size(13)

    # Make y labels bold
    for label in g.get_yticklabels():
        label.set_fontweight("bold")
        label.set_size(13)

    store(
        folder,
        prefix="stack",
        models=models,
        top_k=top_k,
        target_attr=target_attr,
        min_hardness=min_hardness,
        min_certainty=min_certainty,
        max_hardness=max_hardness,
        max_certainty=max_certainty,
        df=df,
    )


def plot_attr(
    path: str,
    min_hardness: int = 0,
    min_certainty: int = 0,
    max_hardness: int = 5,
    max_certainty: int = 5,
    top_k: int = 1,
    target_attr: List[str] = ["all"],
    folder: str = "attr",
    models: List[str] = [],
    show_less_precise: bool = False,
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
        max_hardness=max_hardness,
        max_certainty=max_certainty,
    )

    models_sum = df.groupby("model")["sum_value"].sum()
    sorted_models = models_sum.sort_values(ascending=True).index.tolist()

    if len(sorted_models) > 3:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure(figsize=(7, 5))
    sns.set_theme(style="darkgrid")
    sns.set_palette("muted")

    g = sns.barplot(
        x="sum_value",
        y="attribute",
        orient="h",
        order=sorted_attributes,
        palette=colors,
        data=df,
        errorbar="ci",
        errcolor="grey",
    )

    plt.xlim(0.3, 1)
    for label in g.get_yticklabels():
        label.set_fontweight("bold")

    for label in g.get_xticklabels():
        label.set_fontweight("bold")
    # Set legend

    plt.xlabel("Accuracy", fontdict={"weight": "bold"})
    plt.ylabel("")
    plt.tight_layout()

    store(
        folder=folder,
        prefix="attr",
        models=models,
        top_k=top_k,
        target_attr=target_attr,
        min_hardness=min_hardness,
        min_certainty=min_certainty,
        max_hardness=max_hardness,
        max_certainty=max_certainty,
        df=df,
    )


def plot_drop(
    paths: List[str],
    min_hardness: int = 0,
    min_certainty: int = 0,
    max_hardness: int = 5,
    max_certainty: int = 5,
    top_k: int = 1,
    target_attr: str = "all",
    folder: str = "full",
    models: List[str] = [],
    show_less_precise: bool = False,
):
    assert len(paths) == 2
    base_profiles = load_data(paths[0])
    drop_profiles = load_data(paths[1])

    target_indices = []
    sorted_targets = []
    if target_attr != "all":
        for idx, attribute in enumerate(sorted_attributes):
            if attribute in target_attr:
                target_indices.append(idx)
                sorted_targets.append(attribute)

        loc_colors = [colors[i] for i in target_indices]
    else:
        loc_colors = colors

    loc_colors = loc_colors

    base_df = profiles_to_df(base_profiles)
    drop_df = profiles_to_df(drop_profiles)

    base_df = filter_and_format_df(
        base_df,
        min_hardness,
        min_certainty,
        top_k,
        target_attr,
        models,
        max_hardness=max_hardness,
        max_certainty=max_certainty,
    )
    drop_df = filter_and_format_df(
        drop_df,
        min_hardness,
        min_certainty,
        top_k,
        target_attr,
        models,
        max_hardness=max_hardness,
        max_certainty=max_certainty,
    )

    models_sum = base_df.groupby("model")["sum_value"].sum()
    sorted_models = models_sum.sort_values(ascending=True).index.tolist()

    # Use hue
    if len(sorted_models) > 3:
        plt.figure(figsize=(15, 7))
    else:
        plt.figure(figsize=(10, 7))
    sns.set_theme(style="darkgrid")
    # sns.set_palette("pastel")

    # Combine the two dataframes, giving the drop_df a "drop" = True column
    base_df["drop"] = False
    drop_df["drop"] = True

    df = pd.concat([base_df, drop_df])

    df["attribute"] = df["attribute"].replace({"pobp": "PoB"})

    g = sns.barplot(
        x="attribute",  # "attribute"
        y="sum_value",
        hue="drop",
        data=df,
        order=["location", "age", "occupation", "PoB", "income"],
        errorbar=None,
        palette=loc_colors,
        errcolor="grey",
        errwidth=0.5,
    )
    plt.legend([], [], frameon=False)

    q, r = g.get_legend_handles_labels()
    legend = plt.legend(
        q[-5:],
        r[-5:],
        frameon=True,
        title="Anonymized",
        title_fontsize="large",
        ncol=5,
        loc="upper right",
        prop={"size": 13, "weight": "bold"},
    )

    legend.legend_handles[0].set_color(loc_colors[0])
    legend.legend_handles[0].set_alpha(0.3)
    legend.legend_handles[1].set_color(loc_colors[0])

    plt.xlabel("")
    plt.ylabel("Accuracy", fontdict={"weight": "bold", "size": 18})

    # Give right colors

    local_colors = [color for color in loc_colors for _ in range(len(paths))]

    x_locs = sorted([(patch.get_x(), patch) for patch in g.patches])

    i = 0
    for (x_loc, patch), color in zip(x_locs, local_colors):
        if i % len(paths) == 0:
            patch.set_color(color)
            patch.set_alpha(0.3)
        else:
            patch.set_color(color)
            patch.set_alpha(1)
        i += 1

    # Make ticks bold
    for label in g.get_xticklabels():
        label.set_fontweight("bold")
        label.set_size(20)

    for label in g.get_yticklabels():
        label.set_fontweight("bold")
        label.set_size(18)

    # plt.show()
    plt.tight_layout()

    # base_df.groupby("hardness")["sum_value"].mean()
    # drop_df.groupby("hardness")["sum_value"].mean()

    store(
        folder=folder,
        prefix="drop_hardness",
        models=models,
        top_k=top_k,
        target_attr=target_attr,
        min_hardness=min_hardness,
        min_certainty=min_certainty,
        max_hardness=max_hardness,
        max_certainty=max_certainty,
        df=df,
    )


def worker(params):
    (
        fn,
        path,
        min_hardness,
        max_hardness,
        min_certainty,
        max_certainty,
        top_k,
        attribute,
        folder,
        models,
        show_less_precise,
    ) = params
    fn(
        path=path,
        min_hardness=min_hardness,
        min_certainty=min_certainty,
        max_hardness=max_hardness,
        max_certainty=max_certainty,
        top_k=top_k,
        target_attr=attribute,
        folder=folder,
        models=models,
        show_less_precise=show_less_precise,
    )
    print(
        f"Done {min_hardness} {max_hardness} {min_certainty} {max_certainty} {top_k} {attribute}"
    )

    return 1


def plot_complete(fn, args):
    params = []

    for min_hardness in [1, 2, 3, 4, 5]:
        rel_max = [min_hardness, 5]
        for max_hardness in rel_max:
            for min_certainty in [1, 3, 5]:
                max_certainty = 5
                for top_k in [1, 2, 3]:
                    for attribute in [
                        "all",
                        "education",
                        "pobp",
                        "location",
                        "age",
                        "income",
                        "gender",
                        "occupation",
                        "married",
                    ]:
                        params.append(
                            (
                                fn,
                                args.path,
                                min_hardness,
                                max_hardness,
                                min_certainty,
                                max_certainty,
                                top_k,
                                attribute,
                                args.folder,
                                args.models,
                                args.show_less_precise,
                            )
                        )

    # create a pool of workers and apply the function to each parameter combination
    pool = Pool()
    pool.map(worker, params)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, nargs="+", default=["data/reddit/results/out_test.jsonl"]
    )
    parser.add_argument("--folder", type=str, default="full")
    parser.add_argument("--show_less_precise", action="store_true")
    parser.add_argument("--models", type=str, nargs="*", default=[])
    parser.add_argument("--attributes", type=str, nargs="*", default=[])
    parser.add_argument("--table", type=str, default="author_aggregated")
    parser.add_argument("--hardness", type=int, default=0)
    parser.add_argument("--certainty", type=int, default=3)
    parser.add_argument("--max_hardness", type=int, default=None)
    parser.add_argument("--max_certainty", type=int, default=None)
    parser.add_argument("--plot_dataset", action="store_true")
    parser.add_argument("--plot_drop", action="store_true")
    parser.add_argument("--plot_stack", action="store_true")
    parser.add_argument("--plot_hardness", action="store_true")
    parser.add_argument("--plot_attributes", action="store_true")
    parser.add_argument("--show_stats", action="store_true")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--seed", type=int, default=123123)
    args, unknown = parser.parse_known_args()

    if len(args.path) == 1:
        args.path = args.path[0]

    if len(args.attributes) == 0:
        args.attributes = ["all"]

    if args.show_stats:
        get_specfic_stats(
            args.path,
            min_hardness=args.hardness,
            min_certainty=args.certainty,
            max_hardness=args.max_hardness,
            max_certainty=args.max_certainty,
            top_k=1,
            target_attr="all",
            folder=args.folder,
            models=args.models,
            show_less_precise=args.show_less_precise,
        )

    if args.plot_drop:
        plot_drop(
            args.path,
            min_hardness=args.hardness,
            min_certainty=args.certainty,
            top_k=1,
            target_attr=["location", "age", "occupation", "pobp", "income"],
            folder=args.folder,
            models=args.models,
            show_less_precise=args.show_less_precise,
        )

    if args.plot_dataset:
        plot_dataset(args.path)
    if args.plot_stack:
        if args.complete:
            plot_complete(plot_stack, args)
        else:
            plot_stack(
                args.path,
                min_hardness=0,
                min_certainty=args.certainty,
                top_k=1,
                target_attr="all",
                folder=args.folder,
                models=args.models,
                show_less_precise=args.show_less_precise,
            )
    if args.plot_hardness:
        if args.complete:
            plot_complete(plot_hardness, args)
        else:
            plot_hardness(
                args.path,
                min_hardness=0,
                min_certainty=3,
                top_k=1,
                target_attr="all",
                folder=args.folder,
                models=args.models,
                show_less_precise=args.show_less_precise,
            )
    if args.plot_attributes:
        if args.complete:
            plot_complete(plot_attr, args)
        else:
            plot_attr(
                args.path,
                min_hardness=0,
                min_certainty=3,
                top_k=1,
                target_attr="all",
                folder=args.folder,
                models=args.models,
                show_less_precise=args.show_less_precise,
            )
