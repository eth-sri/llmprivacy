from typing import List, Optional, Tuple

import folktables
import numpy as np
from folktables import ACSDataSource  # , ACSIncome

from data.acs_dataset import Dataset
from src.acs.folk_constants import get_acs_cat
from src.configs import ACSConfig


def get_ord(year):
    if year.endswith("1"):
        return f"{year}-st"
    if year.endswith("2"):
        return f"{year}-nd"
    if year.endswith("3"):
        return f"{year}-rd"
    return f"{year}-th"


def adult_filter(data):
    """Mimic the filters in place for Adult data.
    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    return df


def cit_filter(feat):
    if get_acs_cat("CIT", feat[12]) == "Born in the U.S.":
        return False
    return True


def get_dataset(task_config: ACSConfig) -> Tuple[Optional[Dataset], List[str], str]:
    print(f"state={task_config.state}")

    ACSIncome = folktables.BasicProblem(
        features=[
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "OCCP",
            "POBP",
            "RELP",
            "WKHP",
            "SEX",
            "RAC1P",
            "PINCP",
            "PUMA",
            "CIT",
            "JWMNP",
            "JWTR",
            "MARHYP",
            "FOD1P",
            "POWPUMA",
        ],
        target="PINCP",
        target_transform=lambda x: x > 50000,
        group="RAC1P",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),  # type: ignore
    )

    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=[task_config.state], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    feat_names = ACSIncome.features
    feat_map = {}
    for i, name in enumerate(feat_names):
        feat_map[name] = i

    selected_ids: List[str] = []

    if task_config.prompt_path is None:
        selected_feats = [feat_map[x] for x in task_config.given_attrs]
        selected_label = feat_map[task_config.target]

        selected_ids = [feat_names[sel] for sel in selected_feats]
        target = feat_names[selected_label]

        reduced_attributes = []

        if task_config.target == "PINCP":
            # Compute median income
            # median_income = np.median(features[:, selected_label])

            features[:, selected_label] = label.astype(np.float32)

        elif task_config.target == "SCHL":

            def mapping(x: int):
                if x < 16:  # No HS
                    return 0.0
                elif x < 21:
                    return 1.0  # HS
                elif x == 21:
                    return 3.0  # Bachelor's
                elif x == 22:
                    return 4.0  # Master's
                elif x == 23:
                    return 5.0  # Professional
                elif x == 24:
                    return 6.0  # Doctorate
                else:
                    assert False

            # Map the features with the function
            for i in range(len(features)):
                features[i, selected_label] = mapping(features[i, selected_label])

            reduced_attributes = ["SCHL"]

        custom_dataset = Dataset(
            features,
            selected_feats,
            label=selected_label,
            data_filter=[cit_filter],
            state=task_config.state,
            reduced_attributes=reduced_attributes,
        )
    else:
        target = task_config.target
        custom_dataset = None

    return custom_dataset, selected_ids, target
