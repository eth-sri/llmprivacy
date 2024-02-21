import argparse
from pandas import DataFrame
from utils.data import load_data
import plotly.graph_objects as go
import plotly.express as px

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/reddit/results/out_test.jsonl"
    )
    parser.add_argument("--table", type=str, default="author_aggregated")
    parser.add_argument("--seed", type=int, default=123123)
    args, unknown = parser.parse_known_args()

    data = load_data(args.path, args)

    indiv_comments = []

    users = {}

    for dp in data:
        dp.pop("comments")
        for subreddit, time in zip(dp["subreddits"], dp["timestamps"]):
            indiv_comments.append(
                {"author": dp["author"], "subreddit": subreddit, "time": time}
            )

        for key in ["time", "timestamp"]:
            for reviewer, review in dp["reviews"].items():
                if key in review:
                    del review[key]

        users[dp["author"]] = dp["reviews"]

    df = DataFrame.from_dict(
        {
            (i, j, k): users[i][j][k]
            for i in users.keys()
            for j in users[i].keys()
            for k in users[i][j].keys()
        },
        orient="index",
    )

    df.index.names = ["author", "reviewer", "attribute"]
    df = df.reset_index()
    filtered_df = df.loc[df["hardness"] > 0]

    fig = px.density_heatmap(
        filtered_df,
        x="attribute",
        y="hardness",
        histfunc="count",
        title="Attribute over hardness Heatmap",
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.show()

    indiv_df = DataFrame(indiv_comments)

    fig = px.histogram(indiv_df, x="subreddit").update_xaxes(
        categoryorder="total descending"
    )
    fig.show()
