import argparse
import os
import json


def load_data(path):
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
    else:
        raise Exception

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="./data/reddit/results/out_final.jsonl"
    )
    parser.add_argument(
        "--outpath", type=str, default="./data/reddit/results/out_final_formatted.jsonl"
    )
    parser.add_argument("--user", type=str, default=os.getlogin())
    args = parser.parse_args()

    data = load_data(args.path)
    out_path = args.outpath

    schema = ["author", "num_comments", "comments", "subreddits", "timestamps"]

    new_lines = []
    print(len(data))

    for dp in data:
        if "reviews" in dp:
            reviewer_keys = [
                k for k in dp["reviews"].keys() if k not in ["time", "timestamp"]
            ]

            for id in ["time", "timestamp"]:
                if id in dp["reviews"]:
                    assert len(reviewer_keys) == 1

                    key = reviewer_keys[0]

                    dp["reviews"][key][id] = dp["reviews"][id]
                    del dp["reviews"][id]

            for key in reviewer_keys:
                if "time" not in dp["reviews"][key]:
                    dp["reviews"][key]["time"] = 0
                    dp["reviews"][key]["timestamp"] = 0

            new_lines.append(dp)
        else:
            if "time" in dp:
                print("Hello")
            new_dp = {}
            new_dp["reviews"] = {}
            new_dp["reviews"][args.user] = {}
            for attr in dp:
                if attr in schema:
                    new_dp[attr] = dp[attr]
                else:
                    new_dp["reviews"][args.user][attr] = dp[attr]

            if "time" not in new_dp["reviews"][args.user]:
                new_dp["reviews"][args.user]["time"] = 0
                new_dp["reviews"][args.user]["timestamp"] = 0

            new_lines.append(new_dp)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for line in new_lines:
            f.write(json.dumps(line) + "\n")
