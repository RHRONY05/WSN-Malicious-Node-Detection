# Explainable Malicious Node Detection in Wireless Sensor Networks (WSN-DS)

Your teacher gave two topics (image-based malware prediction, malicious node prediction). We chose
**malicious node prediction in Wireless Sensor Networks**. It is a networking problem: attacks on the
LEACH routing protocol. It also runs in minutes on a normal laptop CPU, with no GPU and no image processing.

## What the project does (in plain words)
A sensor network is made of many small battery nodes. In each round some nodes become **cluster heads**,
collect data from their neighbours and forward it to the **base station**. An attacker can take over a node:

| Class | What the bad node does |
|---|---|
| Normal | behaves correctly |
| Blackhole | pretends to be a cluster head, then **drops all** packets |
| Grayhole | same, but drops **some** packets (hard to detect) |
| Flooding | sends **lots of advertisement messages** to waste energy |
| TDMA (scheduling) | gives every node the **same time slot**, so packets collide |

The ML model looks at counters for each node (messages sent/received, packets forwarded, energy used)
and predicts which of the 5 classes the node belongs to.

## Folder layout
| Path | What it is |
|---|---|
| `data/WSN-DS.csv` | dataset (374,661 rows, 19 columns, 26 MB) |
| `wsn_ids.py` | full pipeline: preprocessing, feature selection, SMOTE, 7 models, tuning, ensemble, threshold optimisation, confusion matrix, ROC, SHAP, LIME |
| `leakage_check.py` | tests what happens if node-ID columns and duplicates are kept |
| `balancing_study.py` | compares 4 class-balancing strategies (picks the best on validation) |
| `make_methodology.py` | draws the methodology diagram |
| `build_report.py` | builds the final PDF report from the results (**edit team names at the top**) |
| `VIVA_QUESTIONS.md` | 65 viva questions with answers + key numbers |
| `assets/cuet_logo.png` | logo used on the cover |
| `figures/` | all figures used in the report |
| `results/` | all tables (CSV) + `summary.json` |
| `CSE312_WSN_Malicious_Node_Detection_Report.pdf` | **the final report to submit** |

## How to run (Windows / Mac / Linux, or Google Colab / Kaggle)
```bash
pip install scikit-learn imbalanced-learn xgboost lightgbm shap lime pandas seaborn reportlab
python wsn_ids.py
python leakage_check.py
python balancing_study.py
python make_methodology.py
python build_report.py
```
Total time is about 20 minutes on a laptop CPU (most of it is hyper-parameter tuning; later runs reuse results/best_params.json). You need about 50 MB of disk space.

On Colab or Kaggle, upload the `.py` files and `data/WSN-DS.csv` and run the same commands with `!`
in front. To publish the code like the sample report did, upload the folder to GitHub or Kaggle and put the
link in `CODE_LINK` inside `build_report.py`.

## Dataset source
WSN-DS (Almomani et al., Journal of Sensors 2016, doi:10.1155/2016/4731953). It is also on Kaggle as
"WSN-DS". The copy used here came from a public GitHub mirror.
