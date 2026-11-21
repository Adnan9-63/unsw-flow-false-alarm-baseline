import numpy as np
from scipy.stats import beta
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def clopper_pearson_lower(tp: int, fp: int, alpha: float = 0.05) -> float:
    if tp <= 0:
        return 0.0
    return float(beta.ppf(alpha, tp, fp + 1))


def evaluate_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "benign_fp": int(fp),
        "benign_fpr": fpr,
        "cp_lower_95": clopper_pearson_lower(int(tp), int(fp)),
    }
