"""Smoke test: grouped k-fold CV runs and keeps stations out of their own test fold."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def test_groupkfold_no_station_in_own_test_fold():
    # 30 stations, several samples each; the property CV relies on is that a station's
    # rows never appear in both train and test of a fold.
    rows = []
    for s in range(30):
        for v in range(4):
            rows.append({"station_id": f"S{s}", "label": (s + v) % 2, "x": s + v})
    df = pd.DataFrame(rows)
    gkf = GroupKFold(n_splits=5)
    groups = df["station_id"].to_numpy()
    for tr, te in gkf.split(df, df["label"], groups):
        assert not (set(groups[tr]) & set(groups[te]))
        assert len(tr) and len(te)
