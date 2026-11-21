import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

try:
    from scipy.stats import beta
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

FILES = {
    "UNSW_NB15_training-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    ],
    "UNSW_NB15_testing-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
    ],
}

for fname, urls in FILES.items():
    if not os.path.exists(fname):
        print(f"Downloading {fname}...")
        for url in urls:
            try:
                urllib.request.urlretrieve(url, fname)
                print(f"Downloaded {fname} successfully.")
                break
            except Exception as e:
                print(f"Mirror failed ({e}), trying next...")

train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

# Strip ID and multi-class text label so model only sees flow features
drop_cols = [c for c in ["id", "attack_cat", "label"] if c in train_df.columns]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df["label"].values
X_test = test_df.drop(columns=drop_cols)
y_test = test_df["label"].values

cat_cols = [c for c in ["proto", "service", "state"] if c in X_train.columns]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Step A: Fit Isolation Forest ONLY on benign (label == 0) flows to score normality
benign_train = X_train_proc[y_train == 0]
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(benign_train)

train_anom = -iso.score_samples(X_train_proc).reshape(-1, 1)
test_anom = -iso.score_samples(X_test_proc).reshape(-1, 1)

X_train_final = np.hstack([X_train_proc, train_anom])
X_test_final = np.hstack([X_test_proc, test_anom])

# Step B: Train GBDT classifier
clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
clf.fit(X_train_final, y_train)
probs = clf.predict_proba(X_test_final)[:, 1]

print(f"\nDataset: UNSW-NB15 | Train flows: {len(y_train)} | Test flows: {len(y_test)}")
print(f"Test Benign: {(y_test == 0).sum()} | Test Attack: {(y_test == 1).sum()}")
print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}\n")

# Step C: Threshold sweep to cut False Positives on Benign flows
header = f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'Benign FP':<12} {'Benign FPR':<12} {'95% CP Low':<10}"
print(header)
print("-" * len(header))

for thresh in [0.50, 0.70, 0.85, 0.92]:
    preds = (probs >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    fpr = (fp / (fp + tn)) * 100
    cp_low = f"{beta.ppf(0.05, tp, fp + 1):.4f}" if (HAS_SCIPY and tp > 0) else "N/A"
    print(f"{thresh:<10.2f} {prec:<10.4f} {rec:<10.4f} {fp:<12d} {fpr:<11.2f}% {cp_low:<10}")

# Step D: Quick shortcut/leakage check on top features (sample 5,000 test rows for speed)
feature_names = cat_cols + num_cols + ["iso_anomaly_score"]
perm = permutation_importance(clf, X_test_final[:5000], y_test[:5000], n_repeats=3, random_state=42, n_jobs=-1)
top_idx = np.argsort(perm.importances_mean)[::-1][:5]
print("\nTop 5 Most Influential Features (Shortcut/Leakage Check):")
for idx in top_idx:
    print(f"  - {feature_names[idx]}: {perm.importances_mean[idx]:.4f}")