import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fieldvision.model.county import compare_county_shares, county_silage_share, reference_share
from fieldvision.model.evaluate_adapted import fit_threshold, pick_calibration
from fieldvision.model.metrics import (block_bootstrap_ci, classification_metrics,
                                       precision_at_prevalence, weighted_auc)
from fieldvision.model.splits import (grid_block_id, holdout_mask, leave_one_group_out,
                                      spatial_folds)


def test_metrics_on_hand_made_confusion_matrix():
    # 3 TP, 1 FN, 2 FP, 4 TN
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
    m = classification_metrics(y, p)
    assert m["confusion"] == {"tp": 3, "fp": 2, "fn": 1, "tn": 4}
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(2 / 6)
    assert m["f1"] == pytest.approx(2 * 3 / (2 * 3 + 2 + 1))
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["balanced_accuracy"] == pytest.approx(0.5 * (3 / 4 + 4 / 6))


def test_undefined_precision_is_nan_not_zero():
    m = classification_metrics([1, 0, 0], [0, 0, 0])
    assert np.isnan(m["precision"])
    assert m["recall"] == 0 and m["fpr"] == 0


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) + 0.4 * y, 1)          # many ties
    roc, ap = weighted_auc(y, s)
    assert roc == pytest.approx(roc_auc_score(y, s))
    assert ap == pytest.approx(average_precision_score(y, s))


def test_block_bootstrap_interval_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.random(400) * 0.5, 0, 1)
    pred = (p >= 0.5).astype(int)
    blocks = rng.integers(0, 30, 400)
    ci = block_bootstrap_ci(y, pred, p, blocks, n_boot=300)
    point = classification_metrics(y, pred, p)
    for k in ("precision", "recall", "fpr", "roc_auc"):
        assert ci[k][0] <= point[k] <= ci[k][1]
    assert ci["n_blocks"] == 30


def test_precision_at_prevalence():
    assert precision_at_prevalence(1.0, 0.0, 0.1) == pytest.approx(1.0)
    assert precision_at_prevalence(0.8, 0.2, 0.5) == pytest.approx(0.8)


def test_spatial_folds_never_share_blocks():
    rng = np.random.default_rng(2)
    blocks = rng.choice([f"b{i}" for i in range(23)], 500)
    y = rng.integers(0, 2, 500)
    folds = spatial_folds(blocks, y, n_splits=5, seed=0)
    assert len(folds) == 5
    seen = np.zeros(500, dtype=int)
    for tr, te in folds:
        assert not set(blocks[tr]) & set(blocks[te])
        seen[te] += 1
    assert (seen == 1).all()


def test_leave_one_group_out_and_block_exclusion():
    groups = np.array(["2022"] * 4 + ["2023"] * 4)
    blocks = np.array(["a", "b", "c", "d", "a", "b", "e", "f"])
    out = leave_one_group_out(groups)
    assert [g for g, *_ in out] == ["2022", "2023"]
    strict = leave_one_group_out(groups, blocks, exclude_test_blocks=True)
    g, tr, te, dropped = strict[1]
    assert g == "2023" and dropped == 2
    assert not set(blocks[tr]) & set(blocks[te])
    assert leave_one_group_out(np.array(["x", "x"])) == []


def test_holdout_patterns_and_grid_blocks():
    b = np.array(["AT-E460N280", "AT-E470N280", "NL-E390N320"])
    assert holdout_mask(b, ["AT-E46*"]).tolist() == [True, False, False]
    assert holdout_mask(b, []).tolist() == [False] * 3
    g = grid_block_id([-76.9, -76.9001, -76.5, np.nan], [39.03, 39.0301, 39.03, 1.0])
    assert g[0] == g[1] and g[0] != g[2] and g[3] == "unknown"


def test_threshold_recalibration_helpers():
    labels = np.array([1] * 30 + [0] * 70)
    idx = pick_calibration(labels, 10, seed=0)
    assert len(idx) == 10 and 1 <= labels[idx].sum() <= 9
    assert (pick_calibration(labels, 10, seed=0) == idx).all()
    thr, bal = fit_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], original=0.5)
    assert 0.2 < thr <= 0.3 and bal == 1.0


def test_county_shares_area_weighted():
    pred = pd.DataFrame({
        "county": ["A", "A", "A", "B", "B"],
        "area_ha": [10.0, 30.0, 60.0, 5.0, 5.0],
        "p_silage": [0.9, 0.2, 0.8, 0.1, 0.6],
        "predicted_label": ["silage", "grain", "silage", "grain", "silage"],
        "low_evidence": [False, False, True, False, False],
    })
    out = county_silage_share(pred, "county").set_index("county")
    assert out.loc["A", "silage_share_confident"] == pytest.approx(10 / 40)
    assert out.loc["A", "silage_share_all"] == pytest.approx(70 / 100)
    assert out.loc["A", "low_evidence_area_share"] == pytest.approx(0.6)
    assert out.loc["A", "silage_share_expected"] == pytest.approx((9 + 6 + 48) / 100)
    assert out.loc["B", "silage_share_confident"] == pytest.approx(0.5)
    ref = reference_share(pd.DataFrame({"county": ["A", "B"], "sil": ["1,000", "(D)"],
                                        "grn": ["3000", "10"]}), "county", "sil", "grn")
    assert ref.set_index("county").loc["A", "reference_silage_share"] == pytest.approx(0.25)
    merged, summ = compare_county_shares(out.reset_index(), ref, "county")
    assert summ["n_counties"] == 1 and summ["mean_abs_error"] == pytest.approx(0.0)
