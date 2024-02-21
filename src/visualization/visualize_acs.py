import pandas as pd
import sys
import os

from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from cmcrameri import cm
import numpy as np
import matplotlib.ticker as mtick

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.initialization import SafeOpen
from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Comment, Profile


color_map = cm.managua
spaced_points = np.linspace(0, 1, 5)
colors = [color_map(i) for i in reversed(spaced_points)]


attributes = [
    ["PUMA", "PINCP", "CIT"],
    ["PUMA", "PINCP", "CIT", "FOD1P"],
    ["PUMA", "PINCP", "MAR", "OCCP"],
    ["PUMA", "MAR", "OCCP", "CIT", "SEX"],
    ["PUMA", "PINCP", "AGEP", "OCCP", "POBP", "WKHP"],
]

target = ["POBP", "RAC1P", "SEX", "PINCP", "SCHL"]
baseline = [0.09, 0.355, 0.511, 0.618, 0.44]
xgb = [0.227, 0.522, 0.768, 0.761, 0.575]
gpt = [0.208, 0.427, 0.743, 0.742, 0.496]

if __name__ == "__main__":
    # Put all the data into a joint dataframe
    df = pd.DataFrame()

    attributes = [",".join(a) for a in attributes]

    df["attributes"] = attributes
    df["target"] = target
    df["baseline"] = baseline
    df["xgb"] = xgb
    df["gpt"] = gpt

    df["baseline"] *= 100
    df["xgb"] *= 100
    df["gpt"] *= 100

    df_melted = df.melt(
        id_vars=["attributes", "target"],
        value_vars=["baseline", "xgb", "gpt"],
        var_name="method",
        value_name="value",
    )

    # For each attribute, get the baseline, xgb and gpt

    print(df_melted)

    order = ["POBP", "RAC1P", "SCHL", "PINCP", "SEX"]

    df_melted["target"] = df_melted["target"].replace(
        {
            "POBP": "Birthplace",
            "RAC1P": "Race",
            "SCHL": "Education",
            "PINCP": "Income",
            "SEX": "Sex",
        }
    )

    order = ["Birthplace", "Race", "Education", "Income", "Sex"]

    sns.set_theme(style="darkgrid")

    g = sns.barplot(
        data=df_melted,
        x="target",
        y="value",
        hue="method",
        order=order,
        hue_order=["baseline", "gpt", "xgb"],
        palette=colors,
        alpha=0.8,
    )

    # No border
    sns.despine(left=True, bottom=True)
    g.spines["top"].set_visible(False)
    g.spines["right"].set_visible(False)

    # MAke labels bold
    plt.setp(g.get_xticklabels(), fontweight="bold", fontsize=14)
    plt.setp(g.get_yticklabels(), fontweight="bold", fontsize=14)

    plt.xlabel("")
    plt.ylabel("Accuracy", fontsize=15, fontweight="bold")

    plt.legend(
        fontsize=14,
    )

    # prop={"size": 14, "weight": "bold"},

    fmt = "%.0f%%"  # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    g.yaxis.set_major_formatter(yticks)

    plt.tight_layout()

    if not os.path.exists("plots/acs"):
        os.makedirs("plots/acs")

    plt.savefig("plots/acs/acs2.png", dpi=300)

    plt.show()
