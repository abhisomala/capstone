"""Tests that splitting does not leak a station or a year across splits."""

import pandas as pd
import pytest

from src.data.splits import (
    site_split,
    temporal_split,
    assert_no_station_leakage,
)


def _synthetic_labels(n_stations=40, visits_per_station=5):
    """Many stations, each visited several times (the near-duplicate leakage risk)."""
    rows = []
    for s in range(n_stations):
        for v in range(visits_per_station):
            year = 2019 + (v % 5)
            rows.append({
                "station_id": f"ST{s:03d}",
                "date": f"{year}-07-{v+1:02d}",
                "year": str(year),
                "label": (s + v) % 2,
                "latitude": 40.0 + s * 0.01,
                "longitude": -83.0 - s * 0.01,
            })
    return pd.DataFrame(rows)


def test_site_split_no_station_leakage():
    df = _synthetic_labels()
    train, val, test = site_split(df, val_fraction=0.15, test_fraction=0.15, seed=42)
    # The whole point: no station id spans two splits.
    assert_no_station_leakage(train, val, test)
    # No rows lost.
    assert len(train) + len(val) + len(test) == len(df)
    # All three splits are non-empty.
    assert len(train) and len(val) and len(test)


def test_site_split_is_deterministic():
    df = _synthetic_labels()
    a = site_split(df, 0.15, 0.15, seed=42)
    b = site_split(df, 0.15, 0.15, seed=42)
    for x, y in zip(a, b):
        assert set(x["station_id"]) == set(y["station_id"])


def test_temporal_split_holds_out_years():
    df = _synthetic_labels()
    train, val, test = temporal_split(df, val_years=[2022], test_years=[2023])
    assert set(test["year"]) == {"2023"}
    assert set(val["year"]) == {"2022"}
    assert "2023" not in set(train["year"])
    assert "2022" not in set(train["year"])


def test_leakage_assertion_fires():
    df = _synthetic_labels(n_stations=4, visits_per_station=2)
    # Deliberately overlap station ST000 between "train" and "test".
    train = df[df["station_id"].isin(["ST000", "ST001"])]
    test = df[df["station_id"].isin(["ST000", "ST002"])]
    val = df[df["station_id"] == "ST003"]
    with pytest.raises(AssertionError):
        assert_no_station_leakage(train, val, test)
