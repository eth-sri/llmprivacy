import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from docarray import BaseDoc
from docarray.typing import NdArray
from hyperdb import HyperDB

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.reddit.reddit import load_data
from data.reddit.utils.data import load_data as db_load_data


class CommentEmbed(BaseDoc):
    comment_str: str = ""
    username_str: str = ""
    embedding: NdArray[384]


def custom_add_documents(db: HyperDB, documents, embedding_function):
    # Adds documents to the HyperDB instance and computes the embeddings
    embeds = []
    for doc in tqdm(documents):
        embeds.append(embedding_function(doc["comment_str"]))

    stacked_embeddings = np.stack(embeds).astype(np.float32)

    if len(db.documents) > 0:
        db.vectors = np.vstack([db.vectors, stacked_embeddings])
        db.documents.extend(documents)
    else:
        db.documents = documents
        db.vectors = stacked_embeddings


def build_db(model, num_values, in_path: str, out_path: str):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    comments = []

    if in_path.endswith(".duckdb"):
        db_args = argparse.Namespace(
            table="author_aggregated",
            seed=456,
        )
        profiles = db_load_data(in_path, db_args)

        for profile in tqdm(profiles):
            for comment in profile["comments"]:
                comments.append(
                    {"comment_str": comment, "username_str": profile["author"]}
                )

    else:
        profiles = load_data(in_path)

        for profile in tqdm(profiles):
            for comment in profile.comments:
                comments.append(
                    {"comment_str": comment.text, "username_str": profile.username}
                )

    if os.path.exists(out_path):
        db = HyperDB([], key="comment_str", embedding_function=model.encode)
        # Load the HyperDB instance from the save file
        db.load(out_path)
        custom_add_documents(db, comments, model.encode)
    else:
        db = HyperDB([], key="comment_str", embedding_function=model.encode)
        custom_add_documents(db, comments, model.encode)
    db.save(out_path)


def query_db(query, model, path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    db = HyperDB([], key="comment_str", embedding_function=model.encode)
    # Load the HyperDB instance from the save file
    db.load(path)

    # Query the HyperDB instance with a text input
    results = db.query(query, top_k=10)

    print("Closest matches:")
    for result in results:
        print(
            f"{result[0]['username_str']}-Sim: {result[1]:.3f}: {result[0]['comment_str']}"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--build_db", action="store_true")
    argparser.add_argument("--query_db", action="store_true")
    argparser.add_argument("--query", type=str, default="I live in australia")
    argparser.add_argument("--db_size", type=int, default=1000000)
    argparser.add_argument("--in_path", type=str, default="eval_results/merged.jsonl")
    argparser.add_argument("--out_path", type=str, default="./vector_db/db.pickle.gz")
    argparser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2"
    )
    argparser.add_argument("--path", type=str, default=None)
    argparser.add_argument("--sentence", type=str, default=None)
    argparser.add_argument("--num_results", type=str, default="gpt2")

    args = argparser.parse_args()

    if args.build_db:
        build_db(args.model, args.db_size, args.in_path, args.out_path)
    elif args.query_db:
        query_db(args.query, args.model, args.out_path)
    else:
        print("Please specify --build_db or --query_db")
