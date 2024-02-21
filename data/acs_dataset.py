import numpy as np
import bisect
import csv
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from src.acs.folk_constants import get_acs_cat, ACSIncome_categories


class Dataset:
    def __init__(
        self,
        all_features,
        selected_features,
        label: int,
        data_filter=[],
        state="NY",
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        reduced_attributes: list[str] = [],
    ) -> None:
        self.pobp_mapping = {}
        self.state_mapping = {}
        self.fod_mapping = {}
        self.reduced_attributes = reduced_attributes
        with open("parsing/pobp.csv") as fin:
            reader = csv.reader(fin, delimiter=",")
            for row in reader:
                code, country = int(row[0]), row[1]
                if code <= 56:
                    self.state_mapping[country[-2:]] = code
                self.pobp_mapping[code] = country

        with open("parsing/fod.csv") as fin:
            reader = csv.reader(fin, delimiter=",")
            for row in reader:
                code, fod = int(row[0]), row[1]
                self.fod_mapping[code] = fod
        with open("parsing/occ_mapping.json", "r") as fin:
            self.occ_map = json.load(fin)
        with open("parsing/puma_mapping.json", "r") as fin:
            self.puma_mapping = json.load(fin)
        self.sorted_codes = sorted(list(map(int, self.occ_map.keys())))

        self.state_code = self.state_mapping[state]

        assert train_size + val_size + test_size <= 1.0

        # Apply filter

        if len(data_filter) > 0:
            filt_feat = []
            for x in all_features:
                keep = True
                for filt in data_filter:
                    if not filt(x):
                        keep = False
                        break
                if keep:
                    filt_feat.append(x)
            all_features = np.asarray(filt_feat)

        X_sel = all_features[:, tuple(selected_features)]
        X_translated_sel = np.asarray(
            [
                np.asarray(list(self._get_translated_feats(feat=x)))[selected_features]
                for x in all_features
            ]
        )
        y_sel = np.ndarray.astype(all_features[:, label], "int64")

        self.label_map = {}
        self.label_rev_map = {}
        for idx, l in enumerate(sorted(np.unique(y_sel))):
            self.label_map[l] = idx
            self.label_rev_map[idx] = l

        y_sel = np.asarray([self.label_map[x] for x in y_sel], dtype=np.int64)

        # Do our own split
        full_idx = np.arange(0, all_features.shape[0] - 1)
        train_idx, rest_idx = train_test_split(
            full_idx, test_size=(val_size + test_size)
        )
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=test_size / (val_size + test_size)
        )

        self.X_train = X_sel[train_idx]
        self.y_train = y_sel[train_idx]
        self.X_val = X_sel[val_idx]
        self.y_val = y_sel[val_idx]
        self.X_test = X_sel[test_idx]
        self.y_test = y_sel[test_idx]

        self.X_train_tr = X_translated_sel[train_idx]
        self.X_val_tr = X_translated_sel[val_idx]
        self.X_test_tr = X_translated_sel[test_idx]

        # Oh Encode - Atm we assume full categorical
        enc = OneHotEncoder(handle_unknown="ignore")
        X_oh = enc.fit_transform(X=X_sel, y=y_sel)

        self.X_train_oh = X_oh[train_idx]
        self.X_val_oh = X_oh[val_idx]
        self.X_test_oh = X_oh[test_idx]

    def _get_code(self, code):
        pos = bisect.bisect_left(self.sorted_codes, code)
        if pos == len(self.sorted_codes):
            pos = pos - 1
        return self.sorted_codes[pos]

    def _get_pobp(self, code):
        return self.pobp_mapping[code]

    def _get_fod(self, code):
        if code == 0:
            return "-"
        return self.fod_mapping[code]

    def _get_puma(self, code):
        return self.puma_mapping[self.state_code].get(str(code), "-")

    def _get_translated_feats(self, feat):
        age = int(feat[0])

        cow = get_acs_cat("COW", feat[1])
        if "SCHL" in self.reduced_attributes:
            school = get_acs_cat("SCHL", feat[2], reduced=True)
        else:
            school = get_acs_cat("SCHL", feat[2])
        mar = get_acs_cat("MAR", feat[3])

        old_code = int(feat[4])
        new_code = self._get_code(old_code)
        occ = self.occ_map[str(new_code)]

        pobp = self._get_pobp(int(feat[5]))
        relp = get_acs_cat("RELP", feat[6])
        wkhp = int(feat[7])
        sex = get_acs_cat("SEX", feat[8])
        race = get_acs_cat("RAC1P", feat[9])
        income = int(feat[10])

        puma_code = int(feat[11])
        puma = self._get_puma(puma_code)

        cit = get_acs_cat("CIT", feat[12])
        travel_time = int(feat[13])

        travel_veh = get_acs_cat("JWTR", feat[14])

        last_married = int(feat[15])

        fod_code = int(feat[16])
        fod = self._get_fod(fod_code)

        pow_puma_code = int(feat[17])
        pow_puma = self._get_puma(pow_puma_code)

        return (
            age,
            cow,
            school,
            mar,
            occ,
            pobp,
            relp,
            wkhp,
            sex,
            race,
            income,
            puma,
            cit,
            travel_time,
            travel_veh,
            last_married,
            fod,
            pow_puma,
        )

    def get_str_from_shifted_code(self, attr_type, code, reduced=False):
        if attr_type == "POBP":
            gt = self._get_pobp(self.label_rev_map[code])
        else:
            gt = self.label_rev_map[code]
            if attr_type in ACSIncome_categories:
                gt = get_acs_cat(attr_type, float(gt), reduced=reduced)

        return gt
