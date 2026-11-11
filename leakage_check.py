"""Checks what happens when node identifiers + duplicates are kept (as many papers do).
Writes results/leakage_check.csv."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "WSN-DS.csv")
df.columns = [c.strip() for c in df.columns]
y = df.pop("Attack type")
clean = df.drop(columns=["id", "who CH"]).assign(_y=y.values).drop_duplicates()
y_clean = clean.pop("_y")
settings = [("IDs + duplicates kept", df, y), ("IDs + duplicates removed (ours)", clean, y_clean)]
models = {"LightGBM": lambda: LGBMClassifier(random_state=42, verbose=-1),
          "Random Forest": lambda: RandomForestClassifier(100, n_jobs=-1, random_state=42)}
rows = []
for sname, X, yy in settings:
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.2, stratify=yy, random_state=42)
    for mname, mk in models.items():
        m = mk().fit(Xtr, ytr); p = m.predict(Xte)
        imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
        rows.append({"Setting": sname, "Model": mname, "Accuracy": accuracy_score(yte, p),
                     "Macro F1": f1_score(yte, p, average="macro"),
                     "Top-3 features": ", ".join(imp.index[:3]),
                     "ID features rank": ", ".join(str(list(imp.index).index(c) + 1) for c in ["id", "who CH"]
                                                   if c in imp.index) or "-"})
        print(rows[-1])
pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "leakage_check.csv", index=False)
