import gradio as gr
import json
import argparse
import os
import time
import random
import tqdm

from datetime import datetime
from presidio_analyzer import AnalyzerEngine

from utils.data import write_data, load_data

analyzer = AnalyzerEngine()
relevant_entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "PERSON", "NRP"]


from subreddits import (
    location,
    gender_females,
    gender_males,
    age_groups,
    occupations,
    pobp,
    married,
    income,
    education_level,
)


data = []
seed = 0
user = ""
timestamp = 0
data_idx = 0
filter = []
outpath = ""
pbar = None
attributes = [
    "location",
    "gender",
    "age",
    "occupation",
    "pobp",
    "married",
    "income",
    "education",
]


def get_new_datapoint_idx():
    global data
    global data_idx
    global filter
    global pbar

    while data_idx < len(data) - 1:
        new_point = data[data_idx]
        is_valid = True

        if "subreddits" in new_point:
            unique_subreddits = set(new_point["subreddits"])
        else:
            unique_subreddits = set()
            for comment in new_point["comments"]:
                unique_subreddits.add(comment["subreddit"])

        # Only relevant dps
        if len(filter) > 0:
            selected_keys = set(filter)
            if len(unique_subreddits.intersection(selected_keys)) == 0:
                is_valid = False

        ret_data_idx = data_idx
        # Increment the data index
        data_idx += 1
        pbar.update(1)

        if is_valid:
            # Save the current state
            with open(".temp", "w") as f:
                f.write(str(seed) + "\n")
                f.write(str(ret_data_idx) + "\n")

            return ret_data_idx

    print("No more data points")
    exit(1)


def save_and_load_new(full_state_box, *args):
    global out_path
    global data
    global user
    global timestamp
    global data_idx

    # Write out current data point
    if len(full_state_box) > 0:
        assert len(args) == 4 * len(attributes)

        curr_data_point = json.loads(full_state_box)

        has_info = False

        curr_reviews = {}
        if "reviews" in curr_data_point:
            curr_reviews = curr_data_point["reviews"]

        if user in curr_reviews:
            print("WARNING - OVERWRITING PREVIOUS REVIEW")
        curr_reviews[user] = {}

        for i, attr in enumerate(attributes):
            estimated_val = args[i]
            dfs_val = args[i + len(attributes)]
            hardness_val = args[i + 2 * len(attributes)]
            certainty_val = args[i + 3 * len(attributes)]

            if hardness_val == -1:
                estimated_val == ""
            if estimated_val is None:
                estimated_val = ""

            if estimated_val is not None and len(estimated_val) > 0:
                has_info = True

            curr_reviews[user][attr] = {
                "estimate": estimated_val.strip().lower(),
                "detect_from_subreddit": dfs_val,
                "hardness": hardness_val,
                "certainty": certainty_val,
            }

        curr_reviews[user]["timestamp"] = time.time()
        curr_reviews[user]["time"] = time.time() - timestamp
        curr_data_point["reviews"] = curr_reviews

        if has_info:
            write_data(out_path, curr_data_point)

    # Load new data point
    new_idx = get_new_datapoint_idx()
    print(f"Loading new data point {new_idx}")
    new_point = data[new_idx]

    if "subreddits" in new_point:
        unique_subreddits = set(new_point["subreddits"])
        unique_subreddits_str = "\n".join(list(unique_subreddits))
    else:
        unique_subreddits = set()
        for comment in new_point["comments"]:
            unique_subreddits.add(comment["subreddit"])
        unique_subreddits_str = "\n".join(list(unique_subreddits))

    if "timestamp" in new_point:
        triplets = list(
            zip(new_point["comments"], new_point["subreddits"], new_point["timestamps"])
        )
    else:
        triplets = list(
            zip(
                [comment["text"] for comment in new_point["comments"]],
                [comment["subreddit"] for comment in new_point["comments"]],
                [int(float(comment["timestamp"])) for comment in new_point["comments"]],
            )
        )

    # Sort by timestamp
    triplets.sort(key=lambda x: (x[1], x[2]))
    formatted_comments = ""
    curr_sub = ""
    for comment in triplets:
        if comment[1] != curr_sub:
            curr_sub = comment[1]
            formatted_comments += f"r/{curr_sub}\n"
        formatted_comments += f"{datetime.fromtimestamp(int(comment[2])).strftime('%d-%m-%Y')}: {comment[0]}\n"

    anonymize_results = ""
    full_json = json.dumps(new_point)

    an_res = analyzer.analyze(
        text=formatted_comments, entities=relevant_entities, language="en"
    )

    for r in an_res:
        anonymize_results += f"{r.entity_type}: {formatted_comments[r.start:r.end]}\n"

    expected_labels = []
    for unique_subreddit in list(unique_subreddits):
        if unique_subreddit in location:
            expected_labels.append(f"Location: {unique_subreddit}")
        elif unique_subreddit in gender_females:
            expected_labels.append("Gender: Female")
        elif unique_subreddit in gender_males:
            expected_labels.append("Gender: Male")
        elif unique_subreddit in age_groups:
            expected_labels.append(f"Age: {age_groups[unique_subreddit]}")
        elif unique_subreddit in occupations:
            expected_labels.append(f"Occupation: {unique_subreddit}")
        elif unique_subreddit in pobp:
            expected_labels.append(f"Place of Birth: {unique_subreddit}")
        elif unique_subreddit in married:
            expected_labels.append(f"Married: {unique_subreddit}")
        elif unique_subreddit in income:
            expected_labels.append(f"Income: {unique_subreddit}")
        elif unique_subreddit in education_level:
            expected_labels.append(f"Education {education_level[unique_subreddit]}")

    # Set the timestamp to meaure how long grading takes
    timestamp = time.time()

    return (
        formatted_comments,
        anonymize_results,
        unique_subreddits_str,
        "\n".join(expected_labels),
        full_json,
        "",
        None,
        "",
        "",
        "",
        None,
        None,
        None,
        *[False] * len(attributes),
        *[0] * len(attributes),
        *[0] * len(attributes),
    )


def main(data, args):
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            name = gr.Textbox(label="Posts", value="")
            anonymized = gr.Textbox(label="Presidio Results")
        with gr.Row(equal_height=True):
            use_subreddits = gr.Textbox(label="Used Subreddits")
            expected_labels = gr.Textbox(label="Expected Labels")

        inputs = []
        dfs_boxes = []
        hardness_sliders = []
        certainty_sliders = []

        with gr.Row(equal_height=True):
            for i, attr in enumerate(attributes):
                with gr.Column():
                    if attr == "location":
                        input = gr.Textbox(
                            label="Location (City / State / Country)",
                            value="",
                            info="Give the closest city / state / country",
                        )
                    elif attr == "gender":
                        input = gr.Radio(
                            label="Sex",
                            choices=["male", "female"],
                            scale=2,
                            info="Sex of the author",
                        )
                    elif attr == "age":
                        input = gr.Textbox(
                            label="Age",
                            value="",
                            info="Age in years, either explicit, e.g. 25, or a range, e.g. 20-30",
                        )
                    elif attr == "occupation":
                        input = gr.Textbox(
                            label="Occupation",
                            value="",
                            info="Brief Occupation Descriptor, e.g. 'Software Engineer'",
                        )
                    elif attr == "pobp":
                        input = gr.Textbox(
                            label="Place of Birth",
                            value="",
                            info="Give the closest city / state / country",
                        )
                    elif attr == "married":
                        input = gr.Radio(
                            label="Marital Status",
                            choices=[
                                "No Relation",
                                "Relation",
                                "Married",
                                "Divorced",
                            ],
                            info="Relationship status of the person",
                        )
                    elif attr == "income":
                        input = gr.Radio(
                            label="Income",
                            choices=[
                                "No",
                                "Low",
                                "Medium",
                                "High",
                                "Very High",
                            ],
                            info="Annual Income - No: No Income\nLow: < 30k\nMedium: 30k - 60k\nHigh: 60k - 150k\nVery High: > 150k",
                        )
                    elif attr == "education":
                        input = gr.Radio(
                            label="Education Level",
                            choices=[
                                "No HS",
                                "In HS",
                                "HS",
                                "In College",
                                "College",
                                "PhD",
                            ],
                            info="Highest level of education.",
                        )
                    else:
                        raise Exception(f"Unknown attribute {attr}")

                    inputs.append(input)
                    # with gr.Row(equal_height=True):
                    slider = gr.Slider(0, 5, label=f"Hardness", step=1)
                    hardness_sliders.append(slider)
                    slider = gr.Slider(0, 5, label=f"Certainty", step=1)
                    certainty_sliders.append(slider)

                    dfs_box = gr.Checkbox(label="Required Subreddit")
                    dfs_boxes.append(dfs_box)

        greet_btn = gr.Button("Next")

        with gr.Accordion(label="Full JSON", open=False):
            hidden_box = gr.Textbox(
                label="JSON",
                value="",
                max_lines=2,
            )

        greet_btn.click(
            fn=save_and_load_new,
            inputs=[
                hidden_box,
                *inputs,
                *dfs_boxes,
                *hardness_sliders,
                *certainty_sliders,
            ],
            outputs=[
                name,
                anonymized,
                use_subreddits,
                expected_labels,
                hidden_box,
                *inputs,
                *dfs_boxes,
                *hardness_sliders,
                *certainty_sliders,
            ],
        )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./data/selected_examples.json")
    parser.add_argument("--table", type=str, default="author_aggregated")
    parser.add_argument("--outpath", type=str, default="./test_out.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user", type=str, default=os.getlogin())
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--subreddit_filter", type=str, nargs="*")
    args = parser.parse_args()

    random.seed(args.seed)
    seed = args.seed
    user = args.user
    data = load_data(args.path, args)
    out_path = args.outpath

    if args.relabel:
        # Filter to only keep profiles that have a review from another user
        new_data = []
        for dp in data:
            if "reviews" in dp:
                if user not in dp["reviews"]:
                    new_data.append(dp)

        data = new_data

        print(f"Relabeling {len(data)} data points")

    if os.path.isfile(".temp"):
        with open(".temp", "r") as f:
            lines = f.readlines()
            old_seed = int(lines[0])
            data_idx = int(lines[1])

        if old_seed != seed:
            data_idx = 0
    else:
        data_idx = 0

    pbar = tqdm.tqdm(total=len(data))
    pbar.update(data_idx - 1)
    # Show the bar with new value
    pbar.refresh()

    # set filter
    if args.subreddit_filter is not None:
        for key in args.subreddit_filter:
            if key == "location":
                filter += location
            elif key == "gender":
                filter += gender_females
                filter += gender_males
            elif key == "age":
                filter += list(age_groups.keys())
            elif key == "occupation":
                filter += occupations
            elif key == "pobp":
                filter += pobp
            elif key == "married":
                filter += married
            elif key == "income":
                filter += income
            elif key == "education":
                filter += list(education_level.keys())
            else:
                raise Exception(f"Unknown filter {key}")

    main(data, args)

    pbar.close()
