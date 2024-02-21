import json 
import duckdb
import os

def load_data(path, args):
    schema_order = ["author", "num_comments", "comments", "subreddits", "timestamps"]

    extension = path.split(".")[-1]
    data = []
    if extension == "json":
        with open(path) as path:
            data = [json.load(path)]
    elif extension == "jsonl":
        with open(path, "r") as json_file:
            json_list = json_file.readlines()

        for json_str in json_list:
            data.append(json.loads(json_str))
    elif extension == "db" or extension == "duckdb":
        con = duckdb.connect(path)
        con.execute("SET threads TO 1")
        con.execute("SET enable_progress_bar=true")
        data = con.execute(
            f"SELECT * FROM {args.table} USING SAMPLE reservoir(50000 ROWS) REPEATABLE ({args.seed})"
        ).fetchall()

        new_data = []
        for row in data:
            new_data.append({k: v for k, v in zip(schema_order, row)})
        data = new_data

    return data



def write_data(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")