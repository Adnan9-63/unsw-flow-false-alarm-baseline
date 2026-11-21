import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from data_loader import load_splits
from metrics import evaluate_threshold


def run_ablation(drop_sttl: bool = True) -> dict:
    train_df, test_df = load_splits()
    drop_cols = [c for c in ["id", "attack_cat", "label"] if c in train_df.columns]
    if drop_sttl and "sttl" in train_df.columns:
        drop_cols.append("sttl")

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["label"].values
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["label"].values

    cat_cols = [c for c in ["proto", "service", "state"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    X_tr = pre.fit_transform(X_train)
    X_te = pre.transform(X_test)

    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_tr[y_train == 0])

    X_tr_aug = np.hstack([X_tr, -iso.score_samples(X_tr).reshape(-1, 1)])
    X_te_aug = np.hstack([X_te, -iso.score_samples(X_te).reshape(-1, 1)])

    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
    clf.fit(X_tr_aug, y_train)
    probs = clf.predict_proba(X_te_aug)[:, 1]

    res = evaluate_threshold(y_test, probs, threshold=0.92)
    res["roc_auc"] = float(roc_auc_score(y_test, probs))
    res["drop_sttl"] = drop_sttl
    return res


if __name__ == "__main__":
    print(run_ablation(drop_sttl=True))
