"""Draws the methodology workflow figure (figures/fig_methodology.png)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 7.4))
ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis("off")

C = {"data": "#dbeafe", "prep": "#fef3c7", "model": "#dcfce7", "eval": "#fce7f3", "xai": "#ede9fe"}
E = {"data": "#2563eb", "prep": "#d97706", "model": "#16a34a", "eval": "#db2777", "xai": "#7c3aed"}


def box(x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=C[kind], ec=E[kind], lw=1.6))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10, weight="bold", color=E[kind])
    ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=8, color="#1f2937", linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.4, mutation_scale=14))


# Row 1 (left -> right)
W, H = 2.55, 1.65
r1 = 5.45
box(0.2, r1, W, H, "1. WSN Simulation Data", "WSN-DS (LEACH, NS-2)\n374,661 node records\n18 features + label\n5 classes", "data")
box(3.15, r1, W, H, "2. Preprocessing", "Drop node identifiers\n(id, who_CH) -> no leakage\nRemove duplicates\nLabel-encode target", "prep")
box(6.1, r1, W, H, "3. Feature Engineering", "Data_Forward_Ratio\nCtrl_Msgs_Sent\nEnergy_per_Packet\n(19 features total)", "prep")
box(9.05, r1, W - .3, H, "4. Stratified Split", "Train 72% / Val 8%\nTest 20% (untouched,\nreal imbalance)", "prep")
for x in (2.75, 5.7, 8.65):
    arrow(x, r1 + H / 2, x + 0.4, r1 + H / 2)

# Row 2 (right -> left)
r2 = 2.95
box(9.05, r2, W - .3, H + .2, "5. Feature Selection", "Mutual Info + Chi-square\n+ RF importance\n-> combined rank\nk chosen by CV macro-F1", "prep")
box(6.1, r2, W, H + .2, "6. Class Balancing", "4 strategies compared:\nnone | class weights |\nSMOTE | under+SMOTE\nbest on Val: class weights", "prep")
box(3.15, r2, W, H + .2, "7. Models + Tuning", "Baselines: NB, LR, DT,\nRF, ET, XGB, LightGBM\nRandomizedSearchCV\n(3-fold, macro-F1)", "model")
box(0.2, r2, W, H + .2, "8. Ensembling", "Weighted soft voting\n(RF + XGB + LGBM)\nClass-wise threshold\noptimisation on Val", "model")
arrow(10.4, r1, 10.4, r2 + H + .2)
for x in (9.05, 6.1, 3.15):
    arrow(x, r2 + (H + .2) / 2, x - 0.4, r2 + (H + .2) / 2)

# Row 3
r3 = 0.35
box(0.2, r3, 3.6, 1.75, "9. Evaluation (test set)", "Accuracy, macro P/R/F1, MCC\nConfusion matrix, ROC-AUC\nBaseline vs improved\nComparison with literature", "eval")
box(4.3, r3, 3.6, 1.75, "10. Explainable AI", "SHAP global importance\nSHAP beeswarm per attack\nSHAP waterfall (one node)\nLIME local explanation", "xai")
box(8.4, r3, 3.4, 1.75, "11. Output", "Detected node state:\nNormal | Blackhole |\nGrayhole | Flooding |\nTDMA (scheduling)", "eval")
arrow(1.475, r2, 1.475, r3 + 1.75)
arrow(3.8, r3 + .875, 4.3, r3 + .875)
arrow(7.9, r3 + .875, 8.4, r3 + .875)

ax.set_title("Proposed methodology: explainable malicious-node detection in WSN", fontsize=13, weight="bold", pad=6)
plt.savefig(OUT / "fig_methodology.png", dpi=170, bbox_inches="tight")
print("saved")
