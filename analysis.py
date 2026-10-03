"""
analysis.py  -  Supplementary analysis for the ML-NIDS project.

Run from the PROJECT ROOT (the folder containing data/, models/, src/):
    python analysis.py

It reads the artefacts your existing pipeline already produced:
    data/X_train.npy, y_train.npy, X_test.npy, y_test.npy
    data/KDDTrain+.csv, data/KDDTest+.csv   (headerless, 43 columns)
    models/logistic_regression.pkl, decision_tree.pkl, random_forest.pkl

It prints everything and also saves results to results/ (created if missing).
Expected run time: about 3-6 minutes (mostly the Random Forest cross-validation).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

os.makedirs("results", exist_ok=True)
OUT = open("results/analysis_results.txt", "w", encoding="utf-8")


def log(text=""):
    print(text)
    OUT.write(str(text) + "\n")


FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

CATEGORY = {}
for name in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
             "udpstorm", "processtable", "mailbomb"]:
    CATEGORY[name] = "DoS"
for name in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]:
    CATEGORY[name] = "Probe"
for name in ["guess_passwd", "ftp_write", "imap", "phf", "multihop",
             "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
             "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named",
             "worm"]:
    CATEGORY[name] = "R2L"
for name in ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
             "xterm", "ps"]:
    CATEGORY[name] = "U2R"
CATEGORY["normal"] = "Normal"

# ------------------------------------------------------------------ load
log("=" * 70)
log("LOADING DATA AND MODELS")
log("=" * 70)
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

raw_train = pd.read_csv("data/KDDTrain+.csv", header=None)
raw_test = pd.read_csv("data/KDDTest+.csv", header=None)
train_labels = raw_train.iloc[:, 41].astype(str).str.strip()
test_labels = raw_test.iloc[:, 41].astype(str).str.strip()

assert len(test_labels) == len(y_test), "Row count mismatch (test)"
assert len(train_labels) == len(y_train), "Row count mismatch (train)"
# Confirm the row order of the raw CSV matches the saved arrays
assert ((test_labels != "normal").astype(int).values == y_test).all(), \
    "Raw test labels do not line up with y_test.npy"
assert ((train_labels != "normal").astype(int).values == y_train).all(), \
    "Raw train labels do not line up with y_train.npy"
log("Raw CSV labels line up with saved arrays. OK")

models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
}

# ------------------------------------------- 1. exact label counts (Table 4.3)
log("\n" + "=" * 70)
log("1. TRAINING SET LABEL COUNTS (for Table 4.3)")
log("=" * 70)
tc = train_labels.value_counts()
tbl = pd.DataFrame({"Count": tc,
                    "Category": [CATEGORY.get(l, "Unknown") for l in tc.index]})
log(tbl.to_string())
tbl.to_csv("results/train_label_counts.csv")
log("\nCategory totals (train):")
log(train_labels.map(lambda l: CATEGORY.get(l, "Unknown")).value_counts().to_string())
log("\nCategory totals (test):")
log(test_labels.map(lambda l: CATEGORY.get(l, "Unknown")).value_counts().to_string())

train_types = set(train_labels.unique())
test_types = set(test_labels.unique())
novel = sorted(test_types - train_types)
log(f"\nAttack types present in KDDTest+ but ABSENT from KDDTrain+ "
    f"({len(novel)}): {novel}")
novel_rows = test_labels.isin(novel).sum()
log(f"Test records belonging to those unseen types: {novel_rows} "
    f"({100 * novel_rows / len(test_labels):.2f}% of test set)")

# ------------------------------- 2. exact metrics + confusion matrices
log("\n" + "=" * 70)
log("2. EXACT TEST-SET METRICS AND CONFUSION MATRICES")
log("=" * 70)
rows = []
preds = {}
for name, m in models.items():
    p = m.predict(X_test)
    preds[name] = p
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, p),
        "Precision": precision_score(y_test, p),
        "Recall": recall_score(y_test, p),
        "F1": f1_score(y_test, p),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "FP rate": fp / (fp + tn),
        "FN rate (miss rate)": fn / (fn + tp),
    })
metrics = pd.DataFrame(rows)
log(metrics.round(4).to_string(index=False))
metrics.to_csv("results/test_metrics.csv", index=False)

# ------------------------------- 3. training accuracy (overfitting check)
log("\n" + "=" * 70)
log("3. TRAINING vs TEST ACCURACY (overfitting check)")
log("=" * 70)
gap = []
for name, m in models.items():
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, preds[name])
    gap.append({"Model": name, "Train accuracy": tr, "Test accuracy": te,
                "Gap": tr - te})
gap = pd.DataFrame(gap)
log(gap.round(4).to_string(index=False))
gap.to_csv("results/train_vs_test.csv", index=False)

# ------------------------------- 4. per-attack-type recall
log("\n" + "=" * 70)
log("4. PER-ATTACK-TYPE RECALL ON KDDTest+ (detected / total)")
log("=" * 70)
per_type = []
for lab in sorted(test_types):
    if lab == "normal":
        continue
    mask = (test_labels == lab).values
    entry = {"Attack type": lab, "Category": CATEGORY.get(lab, "Unknown"),
             "Test count": int(mask.sum()),
             "Train count": int((train_labels == lab).sum())}
    for name in models:
        entry[name + " recall"] = preds[name][mask].mean()
    per_type.append(entry)
per_type = (pd.DataFrame(per_type)
            .sort_values("Test count", ascending=False))
log(per_type.round(4).to_string(index=False))
per_type.to_csv("results/per_attack_type_recall.csv", index=False)

log("\nRECALL BY ATTACK CATEGORY (DoS / Probe / R2L / U2R):")
cat_rows = []
cats = test_labels.map(lambda l: CATEGORY.get(l, "Unknown"))
for c in ["DoS", "Probe", "R2L", "U2R"]:
    mask = (cats == c).values
    entry = {"Category": c, "Test count": int(mask.sum())}
    for name in models:
        entry[name + " recall"] = preds[name][mask].mean()
    cat_rows.append(entry)
cat_df = pd.DataFrame(cat_rows)
log(cat_df.round(4).to_string(index=False))
cat_df.to_csv("results/per_category_recall.csv", index=False)

unseen_mask = test_labels.isin(novel).values
log("\nRecall on attacks of types UNSEEN in training:")
for name in models:
    if unseen_mask.sum():
        log(f"  {name}: {preds[name][unseen_mask].mean():.4f} "
            f"({int(unseen_mask.sum())} records)")
seen_attack = ((y_test == 1) & (~unseen_mask))
log("Recall on attacks of types SEEN in training:")
for name in models:
    log(f"  {name}: {preds[name][seen_attack].mean():.4f} "
        f"({int(seen_attack.sum())} records)")

# ------------------------------- 5. exact feature importances
log("\n" + "=" * 70)
log("5. TOP 15 RANDOM FOREST FEATURE IMPORTANCES (exact, for Table 4.11)")
log("=" * 70)
rf = models["Random Forest"]
imp = pd.DataFrame({"Feature": FEATURES,
                    "Importance": rf.feature_importances_}) \
    .sort_values("Importance", ascending=False).head(15)
imp.insert(0, "Rank", range(1, 16))
log(imp.round(4).to_string(index=False))
imp.to_csv("results/feature_importance_top15.csv", index=False)

# ------------------------------- 6. confidence threshold analysis
log("\n" + "=" * 70)
log("6. RANDOM FOREST: EFFECT OF LOWERING THE ATTACK-PROBABILITY THRESHOLD")
log("=" * 70)
proba = rf.predict_proba(X_test)[:, 1]
thr_rows = []
for t in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1]:
    p = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
    thr_rows.append({"Threshold": t,
                     "Accuracy": accuracy_score(y_test, p),
                     "Precision": precision_score(y_test, p),
                     "Recall": recall_score(y_test, p),
                     "F1": f1_score(y_test, p),
                     "FP": fp, "FN": fn})
thr = pd.DataFrame(thr_rows)
log(thr.round(4).to_string(index=False))
thr.to_csv("results/threshold_analysis.csv", index=False)

# ------------------------------- 7. cross-validation on training file only
log("\n" + "=" * 70)
log("7. 5-FOLD STRATIFIED CROSS-VALIDATION ON KDDTrain+ ONLY")
log("=" * 70)
log("(Same hyperparameters as the main experiments. Takes a few minutes.)")
log("Note: the scaler and encoders were fitted on the full training file, so")
log("there is minor preprocessing leakage across folds; this does not change")
log("the qualitative conclusion for tree-based models.\n")

cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                            class_weight="balanced", n_jobs=-1),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]
cv_rows = []
for name, m in cv_models.items():
    log(f"Cross-validating {name} ...")
    r = cross_validate(m, X_train, y_train, cv=skf, scoring=scoring)
    row = {"Model": name}
    for s in scoring:
        row[s.capitalize() + " mean"] = r["test_" + s].mean()
        row[s.capitalize() + " std"] = r["test_" + s].std()
    cv_rows.append(row)
cv = pd.DataFrame(cv_rows)
log("\n" + cv.round(4).to_string(index=False))
cv.to_csv("results/cross_validation.csv", index=False)

# ------------------------------- 8. side-by-side summary
log("\n" + "=" * 70)
log("8. SUMMARY: CROSS-VALIDATION (train) vs OFFICIAL TEST SET (KDDTest+)")
log("=" * 70)
summ = pd.DataFrame({
    "Model": list(models),
    "CV accuracy (train)": cv["Accuracy mean"].values,
    "KDDTest+ accuracy": metrics["Accuracy"].values,
    "CV recall (train)": cv["Recall mean"].values,
    "KDDTest+ recall": metrics["Recall"].values,
})
log(summ.round(4).to_string(index=False))
summ.to_csv("results/cv_vs_test_summary.csv", index=False)

log("\nDONE. All tables saved in the results/ folder.")
log("Please send me: results/analysis_results.txt (or screenshots of it).")
OUT.close()
