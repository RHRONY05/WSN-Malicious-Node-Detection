"""Ablation: which class-balancing strategy gives the best macro-F1?
Uses the tuned hyper-parameters from results/summary.json and the same splits as wsn_ids.py.
Selection is done on the VALIDATION set; test scores are reported for transparency.
Writes results/balancing_study.csv and figures/fig_balancing_study.png"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent
S = json.loads((ROOT / "results" / "summary.json").read_text())
bp = S["best_params"]
df = pd.read_csv(ROOT / "data" / "WSN-DS.csv"); df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"Expaned Energy": "Expanded_Energy", "who CH": "Who_CH", "Attack type": "Attack_Type"})
df = df.drop(columns=["id", "Who_CH"]).drop_duplicates().reset_index(drop=True)
df["Data_Forward_Ratio"] = df["Data_Sent_To_BS"] / (df["DATA_R"] + 1)
df["Ctrl_Msgs_Sent"] = df["ADV_S"] + df["JOIN_S"] + df["SCH_S"]
df["Energy_per_Packet"] = df["Expanded_Energy"] / (df["DATA_S"] + df["DATA_R"] + 1)
le = LabelEncoder(); y = le.fit_transform(df.pop("Attack_Type")); X = df[S["selected_features"]]
C = list(le.classes_); normal = C.index("Normal")
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=.1, stratify=y_tmp, random_state=SEED)
cnt = np.bincount(y_tr)


def optimise(P, yv, grid=np.linspace(0.5, 3.0, 26)):
    w = np.ones(P.shape[1]); best = f1_score(yv, P.argmax(1), average="macro")
    for _ in range(3):
        for c in range(P.shape[1]):
            for g in grid:
                cand = w.copy(); cand[c] = g
                f = f1_score(yv, (P * cand).argmax(1), average="macro")
                if f > best + 1e-6: best, w = f, cand
    return w


def strat_none():      return X_tr, y_tr, None
def strat_weight():    return X_tr, y_tr, compute_sample_weight({c: (cnt.max() / cnt[c]) ** 0.5 for c in range(len(C))}, y_tr)
def strat_smote_mod():
    return (*SMOTE(sampling_strategy={c: max(cnt[c], 15000) for c in range(len(C)) if c != normal},
                   random_state=SEED).fit_resample(X_tr, y_tr), None)
def strat_hybrid():
    a, b = RandomUnderSampler(sampling_strategy={normal: 60000}, random_state=SEED).fit_resample(X_tr, y_tr)
    return (*SMOTE(sampling_strategy={c: 20000 for c in range(len(C)) if c != normal},
                   random_state=SEED).fit_resample(a, b), None)


STRATS = {"No resampling": strat_none, "Class weights (sqrt-inverse)": strat_weight,
          "Moderate SMOTE (minorities to 15k)": strat_smote_mod, "Hybrid under-sampling + SMOTE": strat_hybrid}
rows, store = [], {}
for sname, fn in STRATS.items():
    Xs, ys, sw = fn()
    models = [RandomForestClassifier(n_jobs=-1, random_state=SEED, **bp["Random Forest"]),
              XGBClassifier(n_jobs=-1, random_state=SEED, tree_method="hist", eval_metric="mlogloss", **bp["XGBoost"]),
              LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, **bp["LightGBM"])]
    Pv, Pt, wts = [], [], []
    for m in models:
        m.fit(Xs, ys, sample_weight=sw)
        pv = m.predict_proba(X_va); Pv.append(pv); Pt.append(m.predict_proba(X_te))
        wts.append(f1_score(y_va, pv.argmax(1), average="macro"))
    Pv = sum(w * p for w, p in zip(wts, Pv)) / sum(wts); Pt = sum(w * p for w, p in zip(wts, Pt)) / sum(wts)
    cw = optimise(Pv, y_va)
    for tag, pv, pt in [("ensemble", Pv, Pt), ("ensemble + threshold opt.", Pv * cw, Pt * cw)]:
        p = pt.argmax(1)
        rows.append({"Balancing strategy": sname, "Variant": tag,
                     "Val macro-F1": f1_score(y_va, pv.argmax(1), average="macro"),
                     "Test accuracy": accuracy_score(y_te, p),
                     "Test precision (macro)": precision_score(y_te, p, average="macro"),
                     "Test recall (macro)": recall_score(y_te, p, average="macro"),
                     "Test macro-F1": f1_score(y_te, p, average="macro")})
        print(rows[-1], flush=True)
    store[sname] = {"class_weights": cw.tolist(), "voting_weights": wts}
R = pd.DataFrame(rows); R.round(4).to_csv(ROOT / "results" / "balancing_study.csv", index=False)
best = R.loc[R["Val macro-F1"].idxmax()]
(ROOT / "results" / "balancing_choice.json").write_text(json.dumps(
    {"best_strategy": best["Balancing strategy"], "best_variant": best["Variant"], "details": store}, indent=2))
print("BEST (by validation):", best["Balancing strategy"], "|", best["Variant"])

piv = R.pivot(index="Balancing strategy", columns="Variant", values="Test macro-F1").loc[list(STRATS)]
ax = piv.plot.bar(figsize=(8, 3.8), color=["#94a3b8", "#16a34a"], rot=12)
for c in ax.containers: ax.bar_label(c, fmt="%.4f", fontsize=7)
ax.set_ylim(piv.values.min() - .01, piv.values.max() + .006); ax.set_ylabel("Test macro-F1")
ax.set_title("Effect of class-balancing strategy (tuned RF+XGB+LGBM soft-voting ensemble)")
ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
plt.savefig(ROOT / "figures" / "fig_balancing_study.png", dpi=130, bbox_inches="tight")
