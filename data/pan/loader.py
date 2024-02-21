import os
import xmltodict
from typing import Dict, List, Any


def load_pan_data(path: str):
    if "2018" in path:
        mapping_path = os.path.join(path, "en.txt")
        folder_path = os.path.join(path, "text")
        # Load mapping data
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping: Dict[str, str] = {}
            for line in f:
                id, label = line.split(":::")
                mapping[id] = label.strip()

        # Load data
        res: List[Dict[str, Any]] = []
        for filename in os.listdir(folder_path):
            f = os.path.join(folder_path, filename)
            # checking if it is a file
            user_id = filename.split(".")[0]

            if os.path.isfile(f):
                new_dict = {}
                new_dict["user"] = user_id
                new_dict["gender"] = mapping[user_id]

                with open(f, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    dict = xmltodict.parse("".join(lines))
                    new_dict["texts"] = dict["author"]["documents"]["document"]

                res.append(new_dict)

        return res

    else:
        res: List[Dict[str, Any]] = []
        for filename in os.listdir(folder_path):
            f = os.path.join(folder_path, filename)
            # checking if it is a file
            user_id = filename.split(".")[0]
