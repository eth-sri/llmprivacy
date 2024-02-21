import gradio as gr
import json
import argparse
import os
import dataclasses

from typing import List, Dict, Any
from utils.data import write_data, load_data


@dataclasses.dataclass
class SynthProfile:
    personality: Dict[str, str]
    feature: str
    hardness: int
    question_asked: str
    response: str
    guess: str
    guess_correctness: Dict[str, List[int]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "personality": self.personality,
                "feature": self.feature,
                "hardness": self.hardness,
                "question_asked": self.question_asked,
                "response": self.response,
                "guess": self.guess,
                "guess_correctness": self.guess_correctness,
            }
        )

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SynthProfile":
        personality = data["personality"]
        feature = data["feature"]
        hardness = data["hardness"]
        question_asked = data["question_asked"]
        response = data["response"]
        guess = data["guess"]
        guess_correctness = data["guess_correctness"]

        return SynthProfile(
            personality,
            feature,
            hardness,
            question_asked,
            response,
            guess,
            guess_correctness,
        )


data: List[SynthProfile] = []
seed = 0
user = ""
timestamp = 0
data_idx = 0
filter = []
outpath = ""
removed_outpath = ""
attributes = [
    "age",
    "sex",
    "city_country",
    "birth_city_country",
    "education",
    "occupation",
    "income",
    "income_level",
    "relationship_status",
]


def get_new_datapoint_idx():
    global data
    global data_idx
    global filter

    ret_data_idx = data_idx

    if ret_data_idx >= len(data):
        print("Out of points")
        exit(0)
    # Increment the data index
    data_idx += 1

    # Save the current state
    with open(".synth_temp", "w") as f:
        f.write(str(seed) + "\n")
        f.write(str(ret_data_idx) + "\n")

        return ret_data_idx


def save_and_load_new(full_state_box, correct):
    global outpath
    global data
    global data_idx

    # Write out current data point
    if len(full_state_box) > 0:
        full_state_box = json.loads(full_state_box)
        rec_profile: SynthProfile = SynthProfile.from_json(full_state_box)

        if correct == "Yes":
            write_data(outpath, dataclasses.asdict(rec_profile))
        else:
            write_data(removed_outpath, dataclasses.asdict(rec_profile))

    # Load new data point
    new_idx = get_new_datapoint_idx()
    print(f"Loading new data point {new_idx}")
    profile: SynthProfile = SynthProfile.from_json(data[new_idx])

    formatted = f"===Question===\n{profile.question_asked}\n\n===Answer==={profile.response}\n\n===Guess=={profile.guess}"

    full_json = profile.to_json()

    feature = profile.feature
    value = profile.personality[feature]
    hardness = profile.hardness

    label_str = f"{feature}: {value}"

    return (formatted, label_str, full_json, "Yes")


def main(data, args):
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            name = gr.Textbox(label="Comment", value="")
        with gr.Row(equal_height=True):
            expected_labels = gr.Textbox(label="Expected Labels")

        with gr.Row(equal_height=True):
            correct = gr.Radio(
                label="Contained",
                choices=[
                    "No",
                    "Yes",
                ],
                info="Whether the comment is interesting or not",
                value="No",
            )

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
                correct,
            ],
            outputs=[
                name,
                expected_labels,
                hidden_box,
                correct,
            ],
        )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="data/synthetic/samples/gpt-4-gpt-4/samples_all_hardness_all_features_0_0_split2.jsonl",
    )
    parser.add_argument("--outpath", type=str, default="./synth_interesting.jsonl")
    parser.add_argument(
        "--removed_outpath", type=str, default="./synth_non_interesting.jsonl"
    )
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    data = load_data(args.path, args)
    outpath = args.outpath
    removed_outpath = args.removed_outpath

    if os.path.isfile(".selected_synth_temp"):
        with open(".selected_synth_temp", "r") as f:
            lines = f.readlines()
            old_seed = int(lines[0])
            data_idx = int(lines[1])

        if old_seed != seed:
            data_idx = 0
    else:
        data_idx = 0

    if args.offset > 0:
        data_idx = max(args.offset, data_idx)

    main(data, args)
