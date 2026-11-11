"""
Malicious Node / DoS Attack Detection in Wireless Sensor Networks (WSN-DS)
==========================================================================
End-to-end pipeline: EDA -> leakage-free preprocessing -> feature engineering ->
ensemble feature selection -> hybrid resampling -> baseline models ->
hyper-parameter tuning -> ensembles -> class-threshold optimisation ->
evaluation (confusion matrix, ROC) -> explainability (SHAP + LIME).

Run:  python wsn_ids.py          (needs data/WSN-DS.csv)
All figures go to figures/, all tables/metrics go to results/.
"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, label_binarize
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "WSN-DS.csv"
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#2563eb", "#dc2626", "#f59e0b", "#16a34a", "#7c3aed", "#0891b2", "#db2777", "#525252"]
LOG = {}
T0 = time.time()


def step(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# 1. Load + dataset glimpse
# ----------------------------------------------------------------------------
step("Loading dataset")
df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"Expaned Energy": "Expanded_Energy", "who CH": "Who_CH",
                        "Attack type": "Attack_Type"})
LOG["raw_shape"] = list(df.shape)
LOG["raw_class_counts"] = df["Attack_Type"].value_counts().to_dict()
LOG["missing_values"] = int(df.isna().sum().sum())
LOG["exact_duplicates"] = int(df.duplicated().sum())
df.head(8).to_csv(RES / "dataset_glimpse.csv", index=False)
df.describe().T.round(3).to_csv(RES / "dataset_describe.csv")

# Class distribution figure
cc = df["Attack_Type"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
bars = ax[0].bar(cc.index, cc.values, color=PALETTE[:5])
ax[0].set_yscale("log"); ax[0].set_ylabel("Records (log scale)")
ax[0].set_title("Class distribution of WSN-DS")
for b, v in zip(bars, cc.values):
    ax[0].text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:,}", ha="center", fontsize=8)
# small slices get no text on the pie; names + shares go in the legend instead
ax[1].pie(cc.values, colors=PALETTE[:5], startangle=90, counterclock=False,
          autopct=lambda v: f"{v:.1f}%" if v > 5 else "", pctdistance=0.6,
          textprops={"fontsize": 9, "color": "white", "weight": "bold"},
          wedgeprops={"linewidth": 1, "edgecolor": "white"})
ax[1].legend([f"{k}  {v:,} ({100 * v / cc.sum():.2f}%)" for k, v in cc.items()], loc="center left",
             bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
ax[1].set_title("Class share")
plt.savefig(FIG / "fig_class_distribution.png"); plt.close()

# ----------------------------------------------------------------------------
# 2. Preprocessing (leakage-free)
# ----------------------------------------------------------------------------
step("Preprocessing")
# 'id' and 'Who_CH' are node identifiers (id encodes round+node number). Keeping
# them lets the model memorise WHICH node was malicious in the simulation instead
# of learning HOW a malicious node behaves -> removed to avoid leakage.
df = df.drop(columns=["id", "Who_CH"])
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
LOG["duplicates_removed_after_id_drop"] = before - len(df)
LOG["clean_shape"] = list(df.shape)
LOG["clean_class_counts"] = df["Attack_Type"].value_counts().to_dict()

# Feature engineering: behaviour ratios that describe routing misbehaviour
df["Data_Forward_Ratio"] = df["Data_Sent_To_BS"] / (df["DATA_R"] + 1)    # CH forwards what it receives?
df["Ctrl_Msgs_Sent"] = df["ADV_S"] + df["JOIN_S"] + df["SCH_S"]          # control-plane chatter (flooding)
df["Energy_per_Packet"] = df["Expanded_Energy"] / (df["DATA_S"] + df["DATA_R"] + 1)
ENGINEERED = ["Data_Forward_Ratio", "Ctrl_Msgs_Sent", "Energy_per_Packet"]

le = LabelEncoder()
y_all = le.fit_transform(df["Attack_Type"])
CLASSES = list(le.classes_)
X_all = df.drop(columns=["Attack_Type"])
FEATURES = list(X_all.columns)
LOG["classes"] = CLASSES
LOG["n_features_total"] = len(FEATURES)

# Correlation heatmap
plt.figure(figsize=(9.5, 7.5))
sns.heatmap(X_all.corr(), cmap="coolwarm", center=0, square=True, linewidths=.3,
            cbar_kws={"shrink": .7}, annot=False)
plt.title("Pearson correlation between features")
plt.savefig(FIG / "fig_correlation.png"); plt.close()

# Behaviour of key features per class
key = ["ADV_S", "SCH_S", "DATA_R", "Data_Sent_To_BS", "Expanded_Energy", "Is_CH"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
means = df.groupby("Attack_Type")[key].mean()
for a, f in zip(axes.ravel(), key):
    a.bar(means.index, means[f], color=PALETTE[:5])
    a.set_title(f"mean {f}", fontsize=10); a.tick_params(axis="x", rotation=30, labelsize=8)
plt.suptitle("Average behaviour of a node by class (why attacks are separable)")
plt.tight_layout(); plt.savefig(FIG / "fig_feature_by_class.png"); plt.close()

# ----------------------------------------------------------------------------
# 3. Split: train / validation / test  (test kept untouched & imbalanced)
# ----------------------------------------------------------------------------
X_tmp, X_test, y_tmp, y_test = train_test_split(X_all, y_all, test_size=0.20,
                                                stratify=y_all, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.10,
                                                  stratify=y_tmp, random_state=SEED)
LOG["split"] = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
step(f"Split train={len(X_train)} val={len(X_val)} test={len(X_test)}")

# ----------------------------------------------------------------------------
# 4. Ensemble feature selection (Mutual Info + Chi2 + RF importance), train only
# ----------------------------------------------------------------------------
step("Feature selection")
rng = np.random.RandomState(SEED)
sub = rng.choice(len(X_train), size=min(60000, len(X_train)), replace=False)
Xs, ys = X_train.iloc[sub], y_train[sub]
mi = mutual_info_classif(Xs, ys, random_state=SEED)
ch, _ = chi2(MinMaxScaler().fit_transform(Xs), ys)
rf_imp = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED,
                                class_weight="balanced").fit(Xs, ys).feature_importances_
norm = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12)
fs = pd.DataFrame({"Feature": FEATURES, "MI": norm(mi), "Chi2": norm(ch), "RF": norm(rf_imp)})
fs["Combined"] = 0.4 * fs["MI"] + 0.2 * fs["Chi2"] + 0.4 * fs["RF"]
fs = fs.sort_values("Combined", ascending=False).reset_index(drop=True)
fs.round(4).to_csv(RES / "feature_scores.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 6))
idx = np.arange(len(fs)); w = 0.27
ax.barh(idx - w, fs["MI"], w, label="Mutual information", color=PALETTE[0])
ax.barh(idx, fs["Chi2"], w, label="Chi-square", color=PALETTE[2])
ax.barh(idx + w, fs["RF"], w, label="RF importance", color=PALETTE[3])
ax.plot(fs["Combined"], idx, "kD", ms=4, label="Combined score")
ax.set_yticks(idx); ax.set_yticklabels(fs["Feature"]); ax.invert_yaxis()
ax.set_xlabel("Normalised score"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("Ensemble feature ranking (training data only)")
plt.savefig(FIG / "fig_feature_selection.png"); plt.close()

# choose k by cross-validated macro-F1
k_grid = [4, 6, 8, 10, 12, 14, len(FEATURES)]
k_scores = []
skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
for k in k_grid:
    cols = fs["Feature"].head(k).tolist()
    sc = []
    for tr, va in skf.split(Xs, ys):
        m = LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1, class_weight="balanced")
        m.fit(Xs.iloc[tr][cols], ys[tr])
        sc.append(f1_score(ys[va], m.predict(Xs.iloc[va][cols]), average="macro"))
    k_scores.append(np.mean(sc))
    step(f"  k={k:2d} macro-F1={k_scores[-1]:.4f}")
best = max(k_scores)
K = min(k for k, s in zip(k_grid, k_scores) if s >= best - 0.002)   # smallest k within 0.2% of best
SEL = fs["Feature"].head(K).tolist()
LOG["k_search"] = dict(zip(map(str, k_grid), [round(s, 4) for s in k_scores]))
LOG["selected_k"] = K; LOG["selected_features"] = SEL
plt.figure(figsize=(6, 3.4))
plt.plot(k_grid, k_scores, "o-", color=PALETTE[0])
plt.axvline(K, ls="--", color=PALETTE[1]); plt.text(K + .3, min(k_scores), f"chosen k={K}", color=PALETTE[1])
plt.xlabel("Number of top-ranked features (k)"); plt.ylabel("3-fold CV macro-F1")
plt.title("Choosing the feature subset size")
plt.savefig(FIG / "fig_k_selection.png"); plt.close()

Xtr, Xva, Xte = X_train[SEL], X_val[SEL], X_test[SEL]

# ----------------------------------------------------------------------------
# 5. Baseline models: imbalanced data, default hyper-parameters
# ----------------------------------------------------------------------------
step("Baseline models")
baseline_models = {
    "Gaussian NB": make_pipeline(StandardScaler(), GaussianNB()),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "XGBoost": XGBClassifier(n_jobs=-1, random_state=SEED, eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1),
}
rows = []
base_pred = {}
for name, m in baseline_models.items():
    t = time.time(); m.fit(Xtr, y_train); p = m.predict(Xte)
    r = metrics(y_test, p); r["Model"] = name; r["Train time (s)"] = time.time() - t
    rows.append(r); base_pred[name] = p
    step(f"  {name:20s} acc={r['Accuracy']:.4f} macroF1={r['F1 (macro)']:.4f}")
baseline_df = pd.DataFrame(rows).set_index("Model")
baseline_df.round(4).to_csv(RES / "baseline_results.csv")

# ----------------------------------------------------------------------------
# 6. Hybrid resampling: under-sample Normal, SMOTE minorities (train only)
# ----------------------------------------------------------------------------
step("Resampling")
normal_id = CLASSES.index("Normal")
counts = np.bincount(y_train)
under = {normal_id: 60000}
over = {c: max(20000, counts[c]) for c in range(len(CLASSES)) if c != normal_id}
X_rs, y_rs = RandomUnderSampler(sampling_strategy=under, random_state=SEED).fit_resample(Xtr, y_train)
X_rs, y_rs = SMOTE(sampling_strategy=over, random_state=SEED, k_neighbors=5).fit_resample(X_rs, y_rs)
LOG["train_before_resample"] = {CLASSES[i]: int(v) for i, v in enumerate(counts)}
LOG["train_after_resample"] = {CLASSES[i]: int(v) for i, v in enumerate(np.bincount(y_rs))}
fig, ax = plt.subplots(figsize=(7, 3.4))
xx = np.arange(len(CLASSES))
ax.bar(xx - .2, counts, .4, label="Before (original train)", color=PALETTE[7])
ax.bar(xx + .2, np.bincount(y_rs), .4, label="After under-sampling + SMOTE", color=PALETTE[0])
ax.set_xticks(xx); ax.set_xticklabels(CLASSES); ax.set_yscale("log"); ax.legend(fontsize=8)
ax.set_ylabel("Training records (log)"); ax.set_title("Training-set balance before vs after resampling")
plt.savefig(FIG / "fig_resampling.png"); plt.close()

# ----------------------------------------------------------------------------
# 7. Hyper-parameter tuning (RandomizedSearchCV, macro-F1)
# ----------------------------------------------------------------------------
step("Tuning")
tsub = rng.choice(len(X_rs), size=min(40000, len(X_rs)), replace=False)
Xt_s, yt_s = X_rs.iloc[tsub] if hasattr(X_rs, "iloc") else X_rs[tsub], y_rs[tsub]
spaces = {
    "Random Forest": (RandomForestClassifier(n_jobs=-1, random_state=SEED), {
        "n_estimators": [150, 250, 400], "max_depth": [None, 20, 30, 40],
        "min_samples_split": [2, 4, 6], "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.5, None]}),
    "XGBoost": (XGBClassifier(n_jobs=-1, random_state=SEED, eval_metric="mlogloss", tree_method="hist"), {
        "n_estimators": [300, 500, 700], "max_depth": [6, 8, 10, 12], "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0], "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 3, 5]}),
    "LightGBM": (LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1), {
        "n_estimators": [300, 500, 800], "num_leaves": [31, 63, 127, 255], "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [-1, 10, 16], "min_child_samples": [10, 20, 40], "subsample": [0.8, 1.0],
        "subsample_freq": [1], "colsample_bytree": [0.7, 0.85, 1.0], "reg_lambda": [0, 1, 5]}),
}
best_params, tune_rows = {}, []
CACHE = RES / "best_params.json"
if CACHE.exists() and (RES / "tuning_results.csv").exists():   # delete best_params.json to re-tune
    best_params = json.loads(CACHE.read_text()); spaces = {}
    step("  using cached tuned parameters (results/best_params.json)")
for name, (est, space) in spaces.items():
    rs = RandomizedSearchCV(est, space, n_iter=12, scoring="f1_macro", cv=3, random_state=SEED,
                            n_jobs=1 if name != "Random Forest" else 2, refit=False)
    rs.fit(Xt_s, yt_s)
    best_params[name] = rs.best_params_
    tune_rows.append({"Model": name, "Best CV macro-F1": rs.best_score_,
                      "Mean CV macro-F1 (all trials)": np.mean(rs.cv_results_["mean_test_score"]),
                      "Best parameters": json.dumps(rs.best_params_)})
    step(f"  {name}: best CV F1={rs.best_score_:.4f} {rs.best_params_}")
if tune_rows:
    pd.DataFrame(tune_rows).to_csv(RES / "tuning_results.csv", index=False)
    CACHE.write_text(json.dumps(best_params, indent=2))
LOG["best_params"] = best_params

# ----------------------------------------------------------------------------
# 8. Improved models (balanced + tuned) and ensembles
# ----------------------------------------------------------------------------
step("Training improved models")
# The balancing ablation (balancing_study.py) compared: no resampling, sqrt-inverse class weights,
# moderate SMOTE and hybrid under-sampling+SMOTE. Class weights won on the VALIDATION set, so the
# final models are trained on the original training data with sqrt-inverse class weights.
SW = {c: float((counts.max() / counts[c]) ** 0.5) for c in range(len(CLASSES))}
LOG["sample_class_weights"] = {CLASSES[c]: round(v, 3) for c, v in SW.items()}
sw_train = np.array([SW[c] for c in y_train])
improved = {
    "RF (tuned+cw)": RandomForestClassifier(n_jobs=-1, random_state=SEED, **best_params["Random Forest"]),
    "XGBoost (tuned+cw)": XGBClassifier(n_jobs=-1, random_state=SEED, eval_metric="mlogloss",
                                        tree_method="hist", **best_params["XGBoost"]),
    "LightGBM (tuned+cw)": LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, **best_params["LightGBM"]),
}
fitted, proba_val, proba_test = {}, {}, {}
for name, m in improved.items():
    t = time.time(); m.fit(Xtr, y_train, sample_weight=sw_train); fitted[name] = m
    proba_val[name] = m.predict_proba(Xva); proba_test[name] = m.predict_proba(Xte)
    step(f"  {name} trained in {time.time() - t:.1f}s")

# Soft-voting ensemble: weights = validation macro-F1 of each member
w = {n: f1_score(y_val, proba_val[n].argmax(1), average="macro") for n in fitted}
wsum = sum(w.values())
proba_val["Soft Voting Ensemble"] = sum(w[n] * proba_val[n] for n in fitted) / wsum
proba_test["Soft Voting Ensemble"] = sum(w[n] * proba_test[n] for n in fitted) / wsum
LOG["voting_weights"] = {k: round(v / wsum, 4) for k, v in w.items()}


# ----------------------------------------------------------------------------
# 9. Class-wise threshold (probability weight) optimisation on VALIDATION set
# ----------------------------------------------------------------------------
def optimise_class_weights(P, y, iters=3, grid=np.linspace(0.5, 3.0, 26)):
    wv = np.ones(P.shape[1])
    best_f = f1_score(y, (P * wv).argmax(1), average="macro")
    for _ in range(iters):
        for c in range(P.shape[1]):
            for g in grid:
                cand = wv.copy(); cand[c] = g
                f = f1_score(y, (P * cand).argmax(1), average="macro")
                if f > best_f + 1e-6:
                    best_f, wv = f, cand
    return wv, best_f


step("Threshold optimisation")
val_f1 = {n: f1_score(y_val, p.argmax(1), average="macro") for n, p in proba_val.items()}
best_name = max(val_f1, key=val_f1.get)
cw, cw_f1 = optimise_class_weights(proba_val[best_name], y_val)
LOG["threshold_base_model"] = best_name
LOG["class_weights"] = {CLASSES[i]: round(float(v), 3) for i, v in enumerate(cw)}
FINAL = f"{best_name} + Threshold Opt."
proba_test[FINAL] = proba_test[best_name] * cw

rows = []
final_pred = {}
for name, P in proba_test.items():
    p = P.argmax(1); final_pred[name] = p
    r = metrics(y_test, p); r["Model"] = name; rows.append(r)
improved_df = pd.DataFrame(rows).set_index("Model")
improved_df.round(4).to_csv(RES / "improved_results.csv")
step("\n" + improved_df.round(4).to_string())

# ----------------------------------------------------------------------------
# 10. Evaluation figures
# ----------------------------------------------------------------------------
step("Evaluation figures")
all_df = pd.concat([baseline_df.drop(columns=["Train time (s)"]).assign(Stage="Baseline"),
                    improved_df.assign(Stage="Improved")])
all_df.round(4).to_csv(RES / "all_results.csv")

fig, ax = plt.subplots(figsize=(11, 4.6))
names = list(all_df.index); x = np.arange(len(names))
ax.bar(x - .2, all_df["Accuracy"], .4, label="Accuracy", color=PALETTE[0])
ax.bar(x + .2, all_df["F1 (macro)"], .4, label="Macro F1-score", color=PALETTE[1])
for i, (a, f) in enumerate(zip(all_df["Accuracy"], all_df["F1 (macro)"])):
    ax.text(i + .2, f + .005, f"{f:.3f}", ha="center", fontsize=7, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
ax.set_ylim(min(0.5, all_df["F1 (macro)"].min() - .05), 1.04)
ax.axvline(len(baseline_df) - .5, color="gray", ls=":")
ax.text(len(baseline_df) / 2 - .5, 1.02, "Baseline (imbalanced, default)", ha="center", fontsize=8)
ax.text(len(baseline_df) + len(improved_df) / 2 - .5, 1.02, "Improved (engineered+tuned+weighted)", ha="center", fontsize=8)
ax.legend(loc="lower left", fontsize=8); ax.set_title("Accuracy vs macro-F1 for all models (test set)")
plt.savefig(FIG / "fig_model_comparison.png"); plt.close()


def plot_cm(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(cmn, annot=np.array([[f"{cmn[i, j] * 100:.1f}%\n({cm[i, j]:,})" for j in range(len(cm))]
                                     for i in range(len(cm))]), fmt="", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, annot_kws={"fontsize": 7.5}, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.savefig(path); plt.close()
    return cm


best_base = baseline_df["F1 (macro)"].idxmax()
LOG["best_baseline"] = best_base
plot_cm(y_test, base_pred[best_base], f"Baseline {best_base}", FIG / "fig_cm_baseline.png")
cm_final = plot_cm(y_test, final_pred[FINAL], "Final model", FIG / "fig_cm_final.png")
pd.DataFrame(cm_final, index=CLASSES, columns=CLASSES).to_csv(RES / "confusion_matrix_final.csv")

rep = classification_report(y_test, final_pred[FINAL], target_names=CLASSES, output_dict=True, digits=4)
pd.DataFrame(rep).T.round(4).to_csv(RES / "classification_report_final.csv")
rep_b = classification_report(y_test, base_pred[best_base], target_names=CLASSES, output_dict=True)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(xx - .2, [rep_b[c]["f1-score"] for c in CLASSES], .4, label=f"Baseline {best_base}", color=PALETTE[7])
ax.bar(xx + .2, [rep[c]["f1-score"] for c in CLASSES], .4, label="Final model", color=PALETTE[3])
for i, c in enumerate(CLASSES):
    ax.text(i + .2, rep[c]["f1-score"] + .005, f"{rep[c]['f1-score']:.3f}", ha="center", fontsize=7)
ax.set_xticks(xx); ax.set_xticklabels(CLASSES); ax.set_ylim(0.7, 1.03); ax.set_ylabel("F1-score")
ax.legend(fontsize=8, loc="lower right"); ax.set_title("Per-class F1: baseline vs final")
plt.savefig(FIG / "fig_per_class_f1.png"); plt.close()

# ROC curves (one-vs-rest)
Yb = label_binarize(y_test, classes=range(len(CLASSES)))
Pf = proba_test[best_name]
plt.figure(figsize=(5.6, 4.4))
LOG["auc"] = {}
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(Yb[:, i], Pf[:, i]); a = auc(fpr, tpr); LOG["auc"][c] = round(a, 5)
    plt.plot(fpr, tpr, color=PALETTE[i], label=f"{c} (AUC={a:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xscale("log"); plt.xlim(1e-4, 1)
plt.xlabel("False positive rate (log)"); plt.ylabel("True positive rate")
plt.legend(fontsize=8, loc="lower right"); plt.title(f"One-vs-rest ROC, {best_name}")
plt.savefig(FIG / "fig_roc.png"); plt.close()

# ----------------------------------------------------------------------------
# 11. Explainability: SHAP (global + local) and LIME (local)
# ----------------------------------------------------------------------------
step("SHAP")
lgbm = fitted["LightGBM (tuned+cw)"]
xs_idx = rng.choice(len(Xte), size=2000, replace=False)
Xshap = Xte.iloc[xs_idx]
explainer = shap.TreeExplainer(lgbm)
sv = explainer.shap_values(Xshap)
if isinstance(sv, list):                          # old API: list of (n, f)
    sv = np.stack(sv, axis=-1)                    # -> (n, f, classes)
mean_abs = np.abs(sv).mean(0)                     # (f, classes)
imp = pd.DataFrame(mean_abs, index=SEL, columns=CLASSES)
imp["Total"] = imp.sum(1); imp = imp.sort_values("Total", ascending=False)
imp.round(4).to_csv(RES / "shap_importance.csv")

fig, ax = plt.subplots(figsize=(8, 4.8))
left = np.zeros(len(imp))
for i, c in enumerate(CLASSES):
    ax.barh(imp.index, imp[c], left=left, color=PALETTE[i], label=c); left += imp[c].values
ax.invert_yaxis(); ax.set_xlabel("mean |SHAP value| (stacked over classes)")
ax.set_title("SHAP global feature importance (LightGBM)"); ax.legend(fontsize=8)
plt.savefig(FIG / "fig_shap_global.png"); plt.close()

for cname in ["Grayhole", "Blackhole", "Flooding"]:
    ci = CLASSES.index(cname)
    plt.figure()
    shap.summary_plot(sv[:, :, ci], Xshap, show=False, max_display=12, plot_size=(7, 4.2))
    plt.title(f"SHAP beeswarm, class = {cname}")
    plt.savefig(FIG / f"fig_shap_beeswarm_{cname.lower()}.png"); plt.close("all")

# local explanation: one correctly-detected grayhole node
gh = CLASSES.index("Grayhole")
pred_shap = lgbm.predict(Xshap)
cand = np.where((y_test[xs_idx] == gh) & (pred_shap == gh))[0]
j = int(cand[0]) if len(cand) else 0
base_val = explainer.expected_value[gh] if np.ndim(explainer.expected_value) else explainer.expected_value
ex = shap.Explanation(values=sv[j, :, gh], base_values=base_val, data=Xshap.iloc[j].values, feature_names=SEL)
plt.figure()
shap.plots.waterfall(ex, max_display=10, show=False)
plt.title("SHAP waterfall: why this node was flagged as Grayhole", fontsize=10)
plt.savefig(FIG / "fig_shap_waterfall.png"); plt.close("all")
LOG["shap_example_row"] = {k: float(v) for k, v in Xshap.iloc[j].items()}

step("LIME")
try:
    from lime.lime_tabular import LimeTabularExplainer
    lime_exp = LimeTabularExplainer(np.asarray(Xtr)[rng.choice(len(Xtr), 20000, replace=False)], feature_names=SEL, class_names=CLASSES,
                                    discretize_continuous=True, random_state=SEED)
    bh = CLASSES.index("Blackhole")
    cand = np.where((y_test == bh) & (final_pred[FINAL] == bh))[0]
    e = lime_exp.explain_instance(Xte.iloc[cand[0]].values, lgbm.predict_proba, num_features=8, labels=[bh])
    lst = e.as_list(label=bh)
    LOG["lime_blackhole"] = lst
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = [v for _, v in lst][::-1]; labs = [k for k, _ in lst][::-1]
    ax.barh(labs, vals, color=[PALETTE[3] if v > 0 else PALETTE[1] for v in vals])
    ax.axvline(0, color="k", lw=.8)
    ax.set_title("LIME: local explanation for a node predicted as Blackhole")
    ax.set_xlabel("Contribution to P(Blackhole)  (green = supports, red = against)")
    plt.savefig(FIG / "fig_lime_blackhole.png"); plt.close()
except Exception as exc:
    LOG["lime_error"] = str(exc)
    step(f"LIME skipped: {exc}")

# ----------------------------------------------------------------------------
# 12. Save summary
# ----------------------------------------------------------------------------
LOG["final_model"] = FINAL
LOG["final_metrics"] = {k: round(float(v), 5) for k, v in improved_df.loc[FINAL].items()}
LOG["best_baseline_metrics"] = {k: round(float(v), 5) for k, v in baseline_df.loc[best_base].items()}
LOG["runtime_sec"] = round(time.time() - T0, 1)
(RES / "summary.json").write_text(json.dumps(LOG, indent=2, default=str))
step("DONE")
