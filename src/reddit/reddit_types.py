from typing import List, Dict, Any, Optional, TextIO
from datetime import datetime
import hashlib
import json


class Comment:
    def __init__(
        self,
        text: str,
        subreddit: str,
        user: str,
        timestamp: str,
        pii: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    ) -> None:
        self.text = text
        self.subreddit = subreddit
        self.user = user
        self.timestamp = datetime.fromtimestamp(int(float(timestamp)))
        self.pii = (
            pii if pii is not None else {}
        )  # model -> type -> "reference", "mentioned" -> str

    def __repr__(self) -> str:
        return f"{self.timestamp.strftime('%Y-%m-%d')}: {self.text}"

    def to_json(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "subreddit": self.subreddit,
            "user": self.user,
            "timestamp": str(self.timestamp.timestamp()),
            "pii": self.pii,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Comment":
        text = data["text"]
        subreddit = data["subreddit"]
        user = data["user"]
        timestamp = data["timestamp"]
        pii = data.get("pii", {})
        return cls(text, subreddit, user, timestamp, pii)

    # Hashable
    def __hash__(self) -> int:
        hash_str = self.text + self.subreddit + self.user + str(self.timestamp)
        return int(hashlib.sha1(hash_str.encode("utf-8")).hexdigest(), 16)

    # Merge two comments
    def __add__(self, other: "Comment") -> "Comment":
        assert self.__hash__() == other.__hash__(), "Comments must be the same"

        # Merge pii dicts
        new_pii = self.pii | other.pii

        return Comment(
            self.text,
            self.subreddit,
            self.user,
            str(self.timestamp.timestamp()),
            new_pii,
        )


class Profile:
    def __init__(
        self,
        username: str,
        comments: List[Comment],
        review_pii: Dict[str, Dict[str, Any]],
        predictions: Optional[Dict[str, Dict[str, Any]]],
        evaluations: Optional[Dict[str, Dict[str, Dict[str, List[int]]]]] = None,
    ) -> None:
        self.username = username
        self.comments = comments
        self.num_comments = len(comments)
        self.review_pii = review_pii
        self.predictions = predictions if predictions is not None else {}
        self.evaluations = (
            evaluations if evaluations is not None else {}
        )  # model -> evaluator -> type -> List[int]
        # Sort comments by subreddit first and timestamp second
        self.comments.sort(key=lambda c: (c.subreddit, c.timestamp))

    def print_review_pii(self):
        for key, value in self.review_pii.items():
            print(f"{key}:")
            if key in ["time", "timestamp"]:
                continue
            for subkey, subvalue in value.items():
                if subkey in ["time", "timestamp"]:
                    continue
                if subvalue["hardness"] > 0:
                    print(
                        f"\t{subkey}: {subvalue['estimate']} - Hardness {subvalue['hardness']} Certainty {subvalue['certainty']}"
                    )

    def get_relevant_pii(self) -> List[str]:
        relevant_pii_type_set: set[str] = set({})

        for reviewer, res in self.review_pii.items():
            if reviewer in ["time", "timestamp"]:
                continue
            for pii_type, pii_res in res.items():
                if pii_type in ["time", "timestamp"]:
                    continue
                else:
                    if pii_res["hardness"] >= 1 and pii_res["certainty"] >= 1:
                        relevant_pii_type_set.add(pii_type)

        relevant_pii_types = list(relevant_pii_type_set)

        return relevant_pii_types

    def to_json(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "comments": [comment.to_json() for comment in self.comments],
            "num_comments": self.num_comments,
            "reviews": self.review_pii,
            "predictions": self.predictions,
            "evaluations": self.evaluations,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Profile":
        username = data["username"]
        comments_json = data["comments"]
        comments = [Comment.from_json(com) for com in comments_json]
        review_pii = data["reviews"]
        predictions = data.get("predictions", {})
        evaluations = data.get("evaluations", {})
        return cls(username, comments, review_pii, predictions, evaluations)

    def to_file(self, file: TextIO) -> None:
        file.write(json.dumps(self.to_json()) + "\n")
        file.flush()

    def print(self, include_attributes: bool = True) -> None:
        print(f"Profile: {self.username} - {self.num_comments} comments")
        print(f"Review PII:")
        self.print_review_pii()
        if include_attributes:
            print(f"Predictions:")
            for model, prediction in self.predictions.items():
                print(f"\t{model}: ")

                for key, value in prediction.items():
                    if key in ["full_answer"]:
                        continue
                    print(f"TYPE: {key}")
                    if "inference" in value:
                        print(f"Inference: {value['inference']}")
                    if "guess" in value:
                        print(f"Guess: {value['guess']}")

                    for evaluator in self.evaluations[model]:
                        if key in self.evaluations[model][evaluator].keys():
                            print(
                                f"EVAL: {evaluator} - {self.evaluations[model][evaluator][key]}"
                            )

    def print_model_response(self, target_model) -> str:
        res_str = f"Profile: {self.num_comments} comments\n"

        comment_str = "\n".join([str(c) for c in self.comments])

        gt_str = ""

        for review in self.review_pii.values():
            for pii_type, pii_res in review.items():
                if pii_type in ["time", "timestamp"]:
                    continue
                if pii_res["hardness"] >= 1 and pii_res["certainty"] >= 1:
                    gt_str += f"{pii_type}: {pii_res['estimate']} - Hardness: {pii_res['hardness']} Certainty: {pii_res['certainty']}\n"

        answer_str = ""

        for model, prediction in self.predictions.items():
            if model != target_model:
                continue

            if "full_answer" in prediction:
                answer_str = "Full answer saved\n" + prediction["full_answer"]
            else:
                answer_str = "NO FULL ANSWER SAVED - REPEATING EXTRACTED ANSWERS\n"
                for key, value in prediction.items():
                    answer_str += f"Key {key}\n"
                    if "inference" in value:
                        answer_str += f"Inference {value['inference']}\n"
                    if "guess" in value:
                        answer_str += f"Guess {value['guess']}\n"

        return (
            res_str
            + comment_str
            + "\nGT:\n\n"
            + gt_str
            + f"\nAnswer - {target_model}\n\n"
            + answer_str
        )

    def __repr__(self) -> str:
        return f"{self.username} - {self.num_comments} comments"

    def __str__(self) -> str:
        res_str = f"{self.username} - {self.num_comments} comments\n"
        for comment in self.comments:
            res_str += f"{comment.subreddit}-{comment.text}\n"
        return res_str

    # Hashable
    def __hash__(self) -> int:
        # Concatenate all comments
        comments_str = "".join([comment.text for comment in self.comments])
        hash_str = self.username  # + comments_str

        return int(hashlib.sha1(hash_str.encode("utf-8")).hexdigest(), 16)

    # Merge two profiles
    def __add__(self, other: "Profile") -> "Profile":
        assert self.__hash__() == other.__hash__(), "Profiles must be the same"

        new_comments = []
        new_review_pii = {}
        new_predictions = {}
        new_evaluations = {}

        for comment1, comment2 in zip(self.comments, other.comments):
            assert (
                comment1.__hash__() == comment2.__hash__()
            ), "Comments must be the same"

            new_comments.append(comment1 + comment2)

        new_review_pii = self.review_pii | other.review_pii
        new_predictions = self.predictions | other.predictions
        new_evaluations = self.evaluations | other.evaluations

        return Profile(
            self.username,
            new_comments,
            new_review_pii,
            new_predictions,
            new_evaluations,
        )
