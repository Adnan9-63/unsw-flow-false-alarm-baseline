import numpy as np
from metrics import clopper_pearson_lower, evaluate_threshold


def test_clopper_pearson_zero_fp():
    bound = clopper_pearson_lower(tp=100, fp=0, alpha=0.05)
    assert 0.95 < bound < 1.0


def test_clopper_pearson_zero_tp():
    assert clopper_pearson_lower(tp=0, fp=5) == 0.0


def test_evaluate_threshold_reduces_fp():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    probs = np.array([0.10, 0.55, 0.72, 0.20, 0.88, 0.95])
    low_t = evaluate_threshold(y_true, probs, threshold=0.50)
    high_t = evaluate_threshold(y_true, probs, threshold=0.85)
    assert high_t["benign_fp"] < low_t["benign_fp"]
    assert high_t["benign_fp"] == 0
