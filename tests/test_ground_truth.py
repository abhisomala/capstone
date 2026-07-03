"""Offline unit tests for the label transforms — no network required."""

import pandas as pd

from src.data.ground_truth import (
    measurement_interval_ugL,
    label_from_interval,
    is_non_detect,
    build_label_frame,
)


def test_unit_conversion_to_ugL():
    assert measurement_interval_ugL(5.0, "ug/L", None)[:2] == (5.0, 5.0)
    assert measurement_interval_ugL(5.0, "ppb", None)[:2] == (5.0, 5.0)
    assert measurement_interval_ugL(0.01, "mg/L", None)[:2] == (10.0, 10.0)  # 0.01 mg/L = 10 µg/L
    assert measurement_interval_ugL(1000.0, "ng/L", None)[:2] == (1.0, 1.0)


def test_unknown_unit_is_dropped():
    lo, hi, note = measurement_interval_ugL(5.0, "furlongs", None)
    assert lo is None and "unknown_unit" in note


def test_non_detect_maps_to_below_threshold():
    assert is_non_detect("Not Detected")
    lo, hi, note = measurement_interval_ugL(None, "ug/L", "Not Detected")
    assert lo == 0.0 and note == "non_detect"


def test_left_censored_is_an_interval():
    # "<0.3 µg/L" means the true value is somewhere in [0, 0.3].
    lo, hi, note = measurement_interval_ugL("<0.3", "ug/L", None)
    assert (lo, hi, note) == (0.0, 0.3, "left_censored")
    assert label_from_interval(lo, hi, 8.0) == 0   # unambiguously below threshold


def test_right_censored_above_threshold():
    lo, hi, note = measurement_interval_ugL(">40", "ug/L", None)
    assert lo == 40.0 and hi == float("inf") and note == "right_censored"
    assert label_from_interval(lo, hi, 8.0) == 1


def test_non_numeric_value_dropped():
    lo, hi, note = measurement_interval_ugL("*", "ug/L", None)
    assert lo is None and note == "non_numeric_value"


def test_label_threshold_and_ambiguity():
    # EPA recreational threshold = 8 µg/L microcystins.
    assert label_from_interval(7.99, 7.99, 8.0) == 0
    assert label_from_interval(8.0, 8.0, 8.0) == 1
    assert label_from_interval(50.0, 50.0, 8.0) == 1
    # "<10" with an 8 µg/L cut straddles the threshold -> unlabelable.
    assert label_from_interval(0.0, 10.0, 8.0) is None


def test_build_label_frame_end_to_end():
    raw = pd.DataFrame([
        # bloom: 12 µg/L >= 8
        {"OrganizationIdentifier": "ORG", "MonitoringLocationIdentifier": "S1",
         "MonitoringLocationName": "Lake A", "ActivityStartDate": "2021-07-01",
         "CharacteristicName": "Microcystin", "ResultMeasureValue": 12.0,
         "ResultMeasure/MeasureUnitCode": "ug/L", "ResultDetectionConditionText": None,
         "ActivityLocation/LatitudeMeasure": 41.5, "ActivityLocation/LongitudeMeasure": -83.2},
        # no-bloom: non-detect -> 0
        {"OrganizationIdentifier": "ORG", "MonitoringLocationIdentifier": "S2",
         "MonitoringLocationName": "Lake B", "ActivityStartDate": "2021-08-15",
         "CharacteristicName": "Microcystin", "ResultMeasureValue": None,
         "ResultMeasure/MeasureUnitCode": "ug/L", "ResultDetectionConditionText": "Not Detected",
         "ActivityLocation/LatitudeMeasure": 27.9, "ActivityLocation/LongitudeMeasure": -81.7},
        # dropped: unknown unit
        {"OrganizationIdentifier": "ORG", "MonitoringLocationIdentifier": "S3",
         "MonitoringLocationName": "Lake C", "ActivityStartDate": "2021-09-01",
         "CharacteristicName": "Microcystin", "ResultMeasureValue": 5.0,
         "ResultMeasure/MeasureUnitCode": "furlongs", "ResultDetectionConditionText": None,
         "ActivityLocation/LatitudeMeasure": 44.0, "ActivityLocation/LongitudeMeasure": -85.0},
    ])
    out = build_label_frame(raw, threshold_ugL=8.0, reference="EPA 2019",
                            source_name="WQP", endpoint="http://x")
    assert len(out) == 2                      # unknown-unit row dropped
    assert out.iloc[0]["label"] == 1
    assert out.iloc[1]["label"] == 0
    assert out.iloc[1]["value_ugL"] == 0.0
    assert out.iloc[0]["year"] == "2021"
    # every retained row carries provenance
    for col in ("source", "station_id", "latitude", "longitude", "date",
                "bloom_threshold_ugL", "threshold_reference"):
        assert out.iloc[0][col] is not None
