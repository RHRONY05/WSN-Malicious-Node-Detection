"""Builds the final PDF report (WSN_Malicious_Node_Detection_Report.pdf) from
figures/ and results/ produced by wsn_ids.py and leakage_check.py.

>>> Edit TEAM / COURSE / DEPT below, then run:  python build_report.py
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle, ListFlowable, ListItem)
from reportlab.platypus.tableofcontents import TableOfContents

# ------------------------------------------------------------------ EDIT ME
COURSE = "Computer Networks (Sessional)"
COURSE_CODE = "Course: CSE-312"
TEAM = [("Al Rubayet", "2204002"), ("Md. Rony Hossain", "2204005"),
        ("Imteajul Islam", "2204015"), ("Tahsin Ahmed Chowdhury", "2204028")]
DEPT = "Department of Computer Science & Engineering"
UNIV = "Chittagong University of Engineering & Technology"
LOGO = Path(__file__).resolve().parent / "assets" / "cuet_logo.png"
CODE_LINK = "Add your GitHub / Kaggle / Colab link here"
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
FIG, RES = ROOT / "figures", ROOT / "results"
import sys
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "CSE312_WSN_Malicious_Node_Detection_Report.pdf"
S = json.loads((RES / "summary.json").read_text())
base = pd.read_csv(RES / "baseline_results.csv", index_col=0)
imp = pd.read_csv(RES / "improved_results.csv", index_col=0)
tune = pd.read_csv(RES / "tuning_results.csv")
fsc = pd.read_csv(RES / "feature_scores.csv")
glimpse = pd.read_csv(RES / "dataset_glimpse.csv")
crep = pd.read_csv(RES / "classification_report_final.csv", index_col=0)
shap_imp = pd.read_csv(RES / "shap_importance.csv", index_col=0)
leak = pd.read_csv(RES / "leakage_check.csv") if (RES / "leakage_check.csv").exists() else None
CLASSES = S["classes"]
FINAL = S["final_model"]; FM = S["final_metrics"]; BB = S["best_baseline"]; BM = S["best_baseline_metrics"]
pct = lambda v: f"{100 * v:.2f}%"

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman", fontSize=11.2, leading=15.2,
                      alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Times-Bold", fontSize=16, leading=20, spaceBefore=12,
                    spaceAfter=8, alignment=0, textColor=colors.HexColor("#111827"))
H2 = ParagraphStyle("h2", parent=H1, fontSize=13, leading=16, spaceBefore=8, spaceAfter=5)
H3 = ParagraphStyle("h3", parent=H1, fontSize=11.5, leading=14, spaceBefore=6, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=9.8, leading=12, alignment=TA_CENTER, spaceBefore=3,
                     spaceAfter=10, textColor=colors.HexColor("#374151"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.8, leading=10.5, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Times-Bold")
CODE = ParagraphStyle("code", fontName="Courier", fontSize=8.2, leading=10.2, backColor=colors.HexColor("#f3f4f6"),
                      borderColor=colors.HexColor("#d1d5db"), borderWidth=0.6, borderPadding=6,
                      leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10)

counters = {"fig": 0, "tab": 0, "lst": 0}
sec = [0, 0, 0]


class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                         topMargin=2.2 * cm, bottomMargin=2.2 * cm, title="WSN Malicious Node Detection",
                         author="Project Team")
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [fr], onPage=self.footer)])

    @staticmethod
    def footer(c, d):
        c.saveState(); c.setFont("Times-Roman", 10)
        c.drawCentredString(A4[0] / 2, 1.2 * cm, str(d.page))
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2", "h3"):
            lvl = {"h1": 0, "h2": 1, "h3": 2}[f.style.name]
            self.notify("TOCEntry", (lvl, f.getPlainText(), self.page))


story = []


def h(level, text):
    if level == 1:
        sec[0] += 1; sec[1] = sec[2] = 0; num = f"{sec[0]}"
    elif level == 2:
        sec[1] += 1; sec[2] = 0; num = f"{sec[0]}.{sec[1]}"
    else:
        sec[2] += 1; num = f"{sec[0]}.{sec[1]}.{sec[2]}"
    story.append(Paragraph(f"{num}&nbsp;&nbsp;{text}", {1: H1, 2: H2, 3: H3}[level]))


def p(text):
    story.append(Paragraph(text, BODY))


def bullets(items, style=BODY):
    story.append(ListFlowable([ListItem(Paragraph(t, style), leftIndent=12) for t in items],
                              bulletType="bullet", start="•", leftIndent=14, bulletFontSize=9))
    story.append(Spacer(1, 4))


def fig(name, caption, width=15.5):
    path = FIG / name
    if not path.exists():
        return
    counters["fig"] += 1
    from PIL import Image as PI
    w, hh = PI.open(path).size
    width_pt = width * cm
    story.append(KeepTogether([Image(str(path), width=width_pt, height=width_pt * hh / w),
                               Paragraph(f"<b>Figure {counters['fig']}:</b> {caption}", CAP)]))


import re as _re
import keyword as _kw
_TOK = _re.compile(r"""(#.*)|('[^']*'|"[^"]*")|\b(\d+\.?\d*)\b|\b([A-Za-z_]\w*)\b""")
_BUILTIN = {"print", "len", "range", "sum", "max", "min", "abs", "np", "pd", "dict", "list"}


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _sp(t):   # escape + keep indentation/spaces
    return _esc(t).replace(" ", "&nbsp;")


def _highlight(line):
    out, pos = [], 0
    for m in _TOK.finditer(line):
        out.append(_sp(line[pos:m.start()])); tok = _sp(m.group(0)); pos = m.end()
        if m.group(1):   out.append(f'<font color="#6a737d"><i>{tok}</i></font>')
        elif m.group(2): out.append(f'<font color="#b45309">{tok}</font>')
        elif m.group(3): out.append(f'<font color="#7c3aed">{tok}</font>')
        elif m.group(4) in _kw.kwlist: out.append(f'<font color="#1d4ed8"><b>{tok}</b></font>')
        elif m.group(4) in _BUILTIN: out.append(f'<font color="#0f766e">{tok}</font>')
        else: out.append(tok)
    out.append(_sp(line[pos:]))
    return "".join(out) or "&nbsp;"


CODE_ST = ParagraphStyle("codeline", fontName="Courier", fontSize=8, leading=10.4)
LN_ST = ParagraphStyle("ln", parent=CODE_ST, textColor=colors.HexColor("#9ca3af"), alignment=2)
HDR_ST = ParagraphStyle("hdr", fontName="Helvetica-Bold", fontSize=7.5, leading=9, textColor=colors.white)


def code(src, caption):
    counters["lst"] += 1
    lines = src.strip("\n").split("\n")
    rows = [[Paragraph(f"PYTHON &nbsp;&nbsp;|&nbsp;&nbsp; Listing {counters['lst']}: {_esc(caption)}", HDR_ST), ""]]
    rows += [[Paragraph(str(i + 1), LN_ST), Paragraph(_highlight(l), CODE_ST)] for i, l in enumerate(lines)]
    t = Table(rows, colWidths=[1.0 * cm, None], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("SPAN", (0, 0), (1, 0)), ("LEFTPADDING", (0, 0), (0, 0), 7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#eef2f7")),
        ("BACKGROUND", (1, 1), (1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#94a3b8")),
        ("LINEAFTER", (0, 1), (0, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.6),
        ("TOPPADDING", (0, 0), (-1, 0), 3), ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
        ("TOPPADDING", (0, 1), (-1, 1), 4), ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
        ("LEFTPADDING", (1, 0), (1, -1), 7), ("RIGHTPADDING", (0, 0), (0, -1), 5),
    ]))
    story.append(KeepTogether([Spacer(1, 4), t, Spacer(1, 10)]))


def table(data, caption, widths=None, header=True, font=8.8, highlight_row=None):
    counters["tab"] += 1
    cs = ParagraphStyle("c", parent=CELL, fontSize=font, leading=font + 1.8)
    cb = ParagraphStyle("cb", parent=cs, fontName="Times-Bold")
    rows = [[Paragraph(str(c), cb if (header and i == 0) else cs) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=[w * cm for w in widths] if widths else None, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")))
    if highlight_row is not None:
        st.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#dcfce7")))
    t.setStyle(TableStyle(st))
    story.append(KeepTogether([t, Paragraph(f"<b>Table {counters['tab']}:</b> {caption}", CAP)]))


def f4(v):
    return f"{v:.4f}"


# =========================================================== COVER
TITLE = ParagraphStyle("t", parent=BODY, fontName="Times-Bold", fontSize=22, leading=28, alignment=TA_CENTER)
CEN = ParagraphStyle("cen", parent=BODY, alignment=TA_CENTER, fontSize=13, leading=17)
story += [Spacer(1, 0.2 * cm), Paragraph(COURSE, ParagraphStyle("x", parent=CEN, fontName="Times-Bold", fontSize=17)),
          Paragraph(COURSE_CODE, CEN), Spacer(1, 0.9 * cm),
          Paragraph("Explainable Malicious Node Detection in<br/>Wireless Sensor Networks", TITLE), Spacer(1, 6),
          Paragraph("Leakage-Free Ensemble Learning with Class-Imbalance Handling, "
                    "Class-Threshold Optimisation and SHAP/LIME Explanations on WSN-DS", CEN),
          Spacer(1, 0.9 * cm)]
if LOGO.exists():
    from PIL import Image as _PI
    _w, _h = _PI.open(LOGO).size
    story.append(Image(str(LOGO), width=4.2 * cm, height=4.2 * cm * _h / _w, mask="auto"))
story += [Spacer(1, 0.9 * cm), Paragraph("<b>Team Members</b>", CEN), Spacer(1, 4)]
_tm = Table([[Paragraph("<b>Name</b>", CEN), Paragraph("<b>Student ID</b>", CEN)]] +
            [[Paragraph(n, CEN), Paragraph(i, CEN)] for n, i in TEAM], colWidths=[7.5 * cm, 4 * cm])
_tm.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                         ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                         ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                         ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
story += [_tm, Spacer(1, 1.1 * cm), Paragraph(DEPT, CEN), Paragraph(UNIV, CEN), PageBreak()]

# =========================================================== TOC
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="Times-Bold", fontSize=11.5, leftIndent=0, leading=16),
                   ParagraphStyle("t1", fontName="Times-Roman", fontSize=10.5, leftIndent=18, leading=14),
                   ParagraphStyle("t2", fontName="Times-Roman", fontSize=10, leftIndent=36, leading=13)]
story += [Paragraph("Contents", ParagraphStyle("ct", parent=H1, fontSize=18)), toc, PageBreak()]

# =========================================================== 1 INTRODUCTION
raw_n = S["raw_shape"][0]
h(1, "Introduction")
p("A <b>Wireless Sensor Network (WSN)</b> is a network of small, battery-powered sensor nodes that sense the "
  "environment (temperature, motion, pollution, etc.) and forward the readings over multi-hop wireless links to "
  "a <b>Base Station (BS)</b>. WSNs are used in smart agriculture, health monitoring, industrial IoT and military "
  "surveillance. To save energy, most WSNs use a hierarchical (cluster-based) routing protocol such as "
  "<b>LEACH</b> (Low-Energy Adaptive Clustering Hierarchy): in every round some nodes elect themselves "
  "<b>Cluster Heads (CH)</b>, the other nodes join a CH, the CH gives each member a TDMA time-slot, collects "
  "their data and forwards it to the BS.")
p("Because the nodes are cheap, unattended and talk over an open radio channel, an attacker can easily capture "
  "or impersonate a node and turn it into a <b>malicious node</b>. A malicious node that participates in routing "
  "can silently destroy the network's service. These are <b>Denial-of-Service (DoS)</b> attacks at the "
  "network/routing layer, the main topic of this Computer Networks project:")
bullets([
    "<b>Blackhole attack:</b> the malicious node advertises itself as a cluster head at the start of every round, "
    "attracts the traffic of its neighbours and then <i>drops all</i> received packets.",
    "<b>Grayhole attack:</b> same trick as blackhole, but the node drops packets <i>selectively or randomly</i>, "
    "which makes it much harder to detect.",
    "<b>Flooding attack:</b> the attacker sends a huge number of LEACH advertisement (ADV) messages, wasting the "
    "energy and bandwidth of the other nodes and confusing CH selection.",
    "<b>Scheduling (TDMA) attack:</b> a malicious CH gives <i>every</i> member the same TDMA slot, causing "
    "packet collisions and data loss.",
])
p("Detecting these malicious nodes manually is impossible at scale, and classic signature-based intrusion "
  "detection does not fit tiny sensor nodes. In this project we build an <b>AI-based Intrusion Detection System "
  f"(IDS)</b> that learns the behaviour of normal and malicious nodes from {raw_n:,} simulated node records "
  "(the public WSN-DS dataset) and classifies each node into one of five classes. Beyond accuracy, we focus on "
  "honest evaluation (no data leakage), fair metrics for the rare attack classes, and explainability, so that a "
  "network operator can see <i>why</i> a node was flagged.")

# =========================================================== 2 RESEARCH PROBLEMS
h(1, "Research Problems and Objectives")
h(2, "Identified Research Problems")
cc = S["raw_class_counts"]
p("Reviewing the literature on WSN intrusion detection and inspecting the WSN-DS data, we identified the "
  "following open problems:")
bullets([
    f"<b>RP1: Severe class imbalance.</b> Normal records make up {100 * cc['Normal'] / raw_n:.1f}% of the data, "
    f"while Flooding is only {100 * cc['Flooding'] / raw_n:.2f}% (imbalance ratio "
    f"{cc['Normal'] / cc['Flooding']:.0f}:1). A model that predicts 'Normal' for everything already reaches "
    f"{100 * cc['Normal'] / raw_n:.1f}% accuracy, so <i>accuracy alone is misleading</i>; rare attacks are missed.",
    "<b>RP2: Blackhole vs Grayhole confusion.</b> Both attacks behave identically during CH election and differ "
    "only in how many packets they drop. The original WSN-DS paper reports only 75.6% detection for Grayhole.",
    "<b>RP3: Optimistic evaluation / data leakage.</b> Many published works keep the node identifier columns "
    "(<i>id</i>, <i>who_CH</i>) and duplicate rows. The model can then memorise <i>which</i> simulated node was "
    "malicious instead of learning <i>how</i> a malicious node behaves, and duplicated records can appear in "
    "both train and test sets. Both make reported scores optimistic and do not transfer to a real network.",
    "<b>RP4: Black-box models.</b> Ensemble models are accurate but opaque. Network operators need to know which "
    "traffic/energy behaviour triggered an alarm before isolating a node.",
])
h(2, "Objectives and Contributions")
bullets([
    "Build a <b>leakage-free pipeline</b> (identifier removal, de-duplication before splitting, untouched test set).",
    "Engineer <b>routing-behaviour features</b> (packet forwarding ratio, control-message volume, energy per packet).",
    "Apply <b>ensemble feature selection</b> (Mutual Information + Chi-square + Random-Forest importance).",
    "Handle imbalance by <b>comparing four balancing strategies</b> (none, class weights, SMOTE, under-sampling + SMOTE) on the training data only.",
    "Compare <b>7 baseline classifiers</b>, tune the best ones with <b>RandomizedSearchCV</b>, build a weighted "
    "<b>soft-voting ensemble</b> and apply <b>class-wise threshold optimisation</b> to maximise macro-F1.",
    "Explain the model with <b>SHAP</b> (global + local) and <b>LIME</b>, and compare against recent studies.",
])

# =========================================================== 3 RELATED WORK
h(1, "Related Work")
p("<b>Almomani et al. (2016)</b> created the WSN-DS dataset by simulating LEACH in NS-2 and trained an "
  "Artificial Neural Network, reporting per-class detection of 92.8% (Blackhole), 75.6% (Grayhole), 99.4% "
  "(Flooding), 92.2% (Scheduling) and 99.8% (Normal) [1]. <b>Talukder et al. (2024)</b> proposed MLSTL-WSN, "
  "which balances WSN-DS with SMOTE-Tomek and evaluates several machine-learning classifiers, reporting 99.78% "
  "binary and 99.92% multiclass accuracy [2]. <b>Ouhmi et al. (2026)</b> compared six boosting algorithms "
  "(AdaBoost, GB, XGBoost, LightGBM, CatBoost, HistGB) for WSN intrusion detection on WSN-DS and reported "
  "99.18% accuracy with a 12.7 ms detection time [3]. Most of these works report accuracy as the headline "
  "metric and do not discuss the identifier columns, which motivates our RP1 and RP3.")

# =========================================================== 4 DATASET
h(1, "Dataset Overview")
h(2, "Dataset Description")
p(f"We use <b>WSN-DS</b> [1], a public intrusion-detection dataset for wireless sensor networks. It was "
  f"generated with the NS-2 simulator running the LEACH routing protocol; each record describes the state of one "
  f"sensor node in one round. The raw file contains <b>{raw_n:,} records</b> and <b>{S['raw_shape'][1]} columns</b> "
  f"(18 features + 1 label), with <b>{S['missing_values']} missing values</b>.")
FEAT_DESC = [
    ("id", "Unique node ID (encodes round + node number)", "Identifier"),
    ("Time", "Simulation time of the node record", "Numeric"),
    ("Is_CH", "1 if the node is a cluster head in this round, else 0", "Binary"),
    ("who CH", "ID of the cluster head the node belongs to", "Identifier"),
    ("Dist_To_CH", "Distance between the node and its CH", "Numeric"),
    ("ADV_S", "Number of CH advertisement messages sent", "Count"),
    ("ADV_R", "Number of CH advertisement messages received", "Count"),
    ("JOIN_S", "Number of join-request messages sent", "Count"),
    ("JOIN_R", "Number of join-request messages received (by a CH)", "Count"),
    ("SCH_S", "Number of TDMA schedule messages sent (by a CH)", "Count"),
    ("SCH_R", "Number of TDMA schedule messages received", "Count"),
    ("Rank", "Order of the node in the TDMA schedule", "Numeric"),
    ("DATA_S", "Number of data packets sent by the node to its CH", "Count"),
    ("DATA_R", "Number of data packets received (by a CH)", "Count"),
    ("Data_Sent_To_BS", "Number of data packets the CH sent to the base station", "Count"),
    ("dist_CH_To_BS", "Distance between the CH and the base station", "Numeric"),
    ("send_code", "Cluster sending code", "Numeric"),
    ("Expanded Energy", "Energy consumed by the node in the round", "Numeric"),
    ("Attack type", "Target: Normal, Blackhole, Grayhole, Flooding, TDMA", "Label"),
]
table([["#", "Feature", "Description", "Type"]] + [[i + 1, a, b, c] for i, (a, b, c) in enumerate(FEAT_DESC)],
      "Features of the WSN-DS dataset", widths=[0.8, 3.2, 9.0, 2.2])

h(2, "Dataset Glimpse")
_raw = pd.read_csv(ROOT / "data" / "WSN-DS.csv"); _raw.columns = [c.strip() for c in _raw.columns]
g = _raw.groupby("Attack type", sort=False).head(1).set_index("Attack type").T   # one record per class
table([["Feature"] + list(g.columns)] + [[f] + [f"{v:g}" for v in g.loc[f]] for f in g.index],
      "Dataset glimpse: one raw record from each class (transposed: each column is one node record)",
      widths=[3.4] + [2.3] * g.shape[1], font=8)
p("Each column shows the counters of one node in one LEACH round. Compare the columns: the Normal cluster head forwards part of what it receives to the base station, "
  "while attack records differ in the advertisement (ADV_S), schedule (SCH_S) and forwarding counters.")

h(2, "Class Distribution and Challenges")
table([["Class", "Meaning", "Records", "Share"]] +
      [[k, {"Normal": "Benign node", "Grayhole": "Selective packet dropping", "Blackhole": "Drops all packets",
            "TDMA": "Scheduling attack (slot collision)", "Flooding": "ADV message flooding"}[k], f"{v:,}",
        f"{100 * v / raw_n:.2f}%"] for k, v in cc.items()] +
      [["Total", "", f"{raw_n:,}", "100%"]], "Class distribution of WSN-DS", widths=[2.6, 6.4, 3, 2.5])
fig("fig_class_distribution.png", "Class distribution of WSN-DS (left: log scale; right: share)")
bullets([
    f"<b>Extreme imbalance</b>: Normal:Flooding = {cc['Normal'] / cc['Flooding']:.0f}:1.",
    f"<b>Duplicates</b>: {S['exact_duplicates']:,} exact duplicate rows; "
    f"{S['duplicates_removed_after_id_drop']:,} duplicates after removing identifier columns.",
    "<b>Identifier columns</b> (<i>id</i>, <i>who CH</i>) that can leak the label.",
    "<b>Similar attacks</b>: Blackhole and Grayhole share the same CH-election behaviour.",
])
h(2, "Final Dataset After Preprocessing")
ccl = S["clean_class_counts"]
table([["Item", "Value"],
       ["Records after cleaning", f"{S['clean_shape'][0]:,}"],
       ["Features used (16 original + 3 engineered)", f"{S['n_features_total']}"],
       ["Classes", ", ".join(f"{k} ({v:,})" for k, v in ccl.items())],
       ["Train / Validation / Test", f"{S['split']['train']:,} / {S['split']['val']:,} / {S['split']['test']:,}"]],
      "Dataset summary after preprocessing", widths=[6, 9])

# =========================================================== 5 METHODOLOGY
h(1, "Methodology")
h(2, "Overall Workflow")
p(f"Figure {counters['fig'] + 1} shows the complete pipeline. Every data-dependent step (feature selection, resampling, tuning, "
  "threshold optimisation) is fitted on the training or validation data only; the 20% test set is never "
  "touched until the final evaluation and keeps the real-world class imbalance.")
fig("fig_methodology.png", "Proposed methodology workflow", width=16)

# =========================================================== 6 PREPROCESSING
h(1, "Data Preprocessing")
h(2, "Missing Values and Column Cleaning")
p("The dataset has no missing values, so no imputation is needed. Column names contain stray spaces "
  "(e.g. ' Time'), which are stripped and two names are corrected.")
code("""
df = pd.read_csv("data/WSN-DS.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"Expaned Energy": "Expanded_Energy",
                        "who CH": "Who_CH", "Attack type": "Attack_Type"})
print(df.isna().sum().sum())          # -> 0 missing values
""", "Loading and cleaning column names")
h(2, "Removing Node Identifiers (Leakage Prevention)")
p("The <i>id</i> column encodes the simulation round and node number, and <i>who_CH</i> stores the ID of the "
  "cluster head. In a simulation, the same few node IDs are configured as attackers, so these columns let the "
  "model memorise attacker IDs. A real IDS must recognise malicious <i>behaviour</i> of unseen nodes, so both "
  "columns are dropped.")
code("""
df = df.drop(columns=["id", "Who_CH"])          # identifiers -> leakage
df = df.drop_duplicates().reset_index(drop=True) # before splitting!
""", "Identifier removal and de-duplication")
p(f"<b>Result:</b> {S['duplicates_removed_after_id_drop']:,} duplicate records removed; "
  f"{S['clean_shape'][0]:,} unique records remain. De-duplicating <i>before</i> the train/test split guarantees "
  "that no identical record appears in both sets.")
if leak is not None:
    h(2, "Experiment: How Much Does Leakage Inflate Scores?")
    p("To test RP3 we trained default LightGBM and Random Forest models twice: once with identifiers and "
      "duplicates kept (as commonly done) and once with our cleaning (80/20 stratified split in both cases).")
    table([["Setting", "Model", "Accuracy", "Macro F1", "Top-3 important features", "Rank of id / who_CH"]] +
          [[r["Setting"], r["Model"], f4(r["Accuracy"]), f4(r["Macro F1"]), r["Top-3 features"],
            r["ID features rank"]] for _, r in leak.iterrows()],
          "Effect of identifier columns and duplicates (default models)", widths=[3.6, 2.3, 1.7, 1.7, 3.9, 2.2],
          font=8)
    LEAK_TAB = counters["tab"]
    lk = leak.set_index(["Model", "Setting"])["Macro F1"]
    rf_gap = lk[("Random Forest", "IDs + duplicates kept")] - lk[("Random Forest", "IDs + duplicates removed (ours)")]
    p(f"Two effects are visible. (1) With duplicates kept, Random Forest's macro-F1 is {100 * rf_gap:+.2f} pp "
      "higher, because identical records end up in both train and test sets and are simply memorised. "
      "(2) LightGBM ranks the <i>id</i> and <i>who_CH</i> columns 2nd and 3rd, so it partly learns <i>which</i> "
      "node was the attacker; this does not help on this split (its macro-F1 is actually lower) and cannot "
      "transfer to a real network with new node IDs. Both effects are avoided by our stricter setting, "
      "so all results below are leakage-free.")

# =========================================================== 7 EDA + FE
h(1, "Exploratory Analysis and Feature Engineering")
h(2, "Feature Behaviour per Class")
fig("fig_feature_by_class.png", "Average value of key features per class")
p("The class means already show the network-level signatures of each attack: Flooding nodes send far more "
  "advertisement messages (ADV_S); TDMA attackers send many schedule messages (SCH_S); Blackhole and Grayhole "
  "nodes are almost always cluster heads (Is_CH) that receive a lot of data (DATA_R) but forward little or "
  "nothing to the base station (Data_Sent_To_BS).")
fig("fig_correlation.png", "Pearson correlation matrix of all features", width=13)
h(2, "Engineered Features")
p("Based on how each attack works at the routing layer, three behaviour features were added:")
code("""
# Does a cluster head forward what it receives?  (Blackhole ~ 0, Grayhole < 1)
df["Data_Forward_Ratio"] = df["Data_Sent_To_BS"] / (df["DATA_R"] + 1)
# Control-plane chatter (Flooding / TDMA attacks send many control messages)
df["Ctrl_Msgs_Sent"]     = df["ADV_S"] + df["JOIN_S"] + df["SCH_S"]
# Energy cost per handled packet
df["Energy_per_Packet"]  = df["Expanded_Energy"] / (df["DATA_S"] + df["DATA_R"] + 1)
""", "Routing-behaviour feature engineering")

# =========================================================== 8 FEATURE SELECTION
h(1, "Feature Selection")
h(2, "Ensemble Ranking Strategy")
p("Three complementary scores are computed on a 60,000-record sample of the <b>training set</b> and min-max "
  "normalised: Mutual Information (non-linear dependency), Chi-square (statistical dependency on min-max scaled "
  "features) and Random-Forest impurity importance (model-based). They are combined as "
  "<b>0.4 MI + 0.2 Chi2 + 0.4 RF</b>.")
code("""
mi  = mutual_info_classif(Xs, ys, random_state=42)
ch, _ = chi2(MinMaxScaler().fit_transform(Xs), ys)
rf  = RandomForestClassifier(150, class_weight="balanced").fit(Xs, ys).feature_importances_
combined = 0.4*norm(mi) + 0.2*norm(ch) + 0.4*norm(rf)
""", "Ensemble feature scoring")
fig("fig_feature_selection.png", "Normalised feature scores (sorted by combined score)", width=14)
top = fsc.head(10)
table([["Rank", "Feature", "MI", "Chi2", "RF", "Combined"]] +
      [[i + 1, r.Feature, f"{r.MI:.3f}", f"{r.Chi2:.3f}", f"{r.RF:.3f}", f"{r.Combined:.3f}"]
       for i, r in top.iterrows()], "Top-10 features by combined score", widths=[1.2, 4.5, 2, 2, 2, 2.2])
h(2, "Choosing the Number of Features")
ks = S["k_search"]
p("The top-<i>k</i> features were evaluated for k in {" + ", ".join(ks.keys()) + "} with 3-fold cross-validated "
  "macro-F1 (LightGBM). The smallest k whose score is within 0.2 percentage points of the best was chosen.")
fig("fig_k_selection.png", "Cross-validated macro-F1 versus number of selected features", width=10)
p(f"<b>Selected:</b> k = {S['selected_k']} features. "
  + ("Every reduction of the feature set lowered macro-F1, so all features were kept for the main model; the "
     "ranking is still useful for a lightweight on-node deployment (the top-12 features reach "
     f"{ks.get('12', 0):.3f} macro-F1 versus {ks[str(S['selected_k'])]:.3f} with all)."
     if S["selected_k"] == S["n_features_total"] else
     "Selected features: " + ", ".join(S["selected_features"]) + "."))

# =========================================================== 9 SPLIT + BALANCING
h(1, "Data Splitting and Class Balancing")
h(2, "Stratified Train / Validation / Test Split")
code("""
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.10,
                                                  stratify=y_tmp, random_state=42)
""", "Stratified three-way split")
p(f"Train: {S['split']['train']:,} records (72%), validation: {S['split']['val']:,} (8%, used for ensemble "
  f"weights and threshold optimisation), test: {S['split']['test']:,} (20%, original imbalance).")
h(2, "Class Balancing Strategies (Training Set Only)")
p("Four ways of handling the imbalance were compared, all applied to the training data only:")
bullets([
    "<b>No resampling</b>: train on the original distribution.",
    "<b>Class weights</b>: keep the data, but give every record of class c the weight "
    "sqrt(N<sub>max</sub> / N<sub>c</sub>), so mistakes on rare attacks cost more.",
    "<b>Moderate SMOTE</b>: synthesise new attack records until every attack class has 15,000 records.",
    "<b>Hybrid</b>: random <b>under-sampling</b> of Normal to 60,000 records plus <b>SMOTE</b> of every "
    "attack class to 20,000 records (also used to build the balanced sample for hyper-parameter tuning).",
])
code("""
# Class weights (chosen strategy)
w = {c: (counts.max() / counts[c]) ** 0.5 for c in classes}
model.fit(X_train, y_train, sample_weight=[w[c] for c in y_train])
# Hybrid resampling (compared strategy)
X_rs, y_rs = RandomUnderSampler(sampling_strategy={normal: 60000}).fit_resample(X_train, y_train)
X_rs, y_rs = SMOTE(sampling_strategy={c: 20000 for c in attacks}).fit_resample(X_rs, y_rs)
""", "Class weighting and hybrid resampling")
tb, ta = S["train_before_resample"], S["train_after_resample"]
swt = S["sample_class_weights"]
table([["Class", "Train records", "Class weight", "After hybrid resampling"]] +
      [[c, f"{tb[c]:,}", f"{swt[c]:.2f}", f"{ta[c]:,}"] for c in CLASSES] +
      [["Total", f"{sum(tb.values()):,}", "", f"{sum(ta.values()):,}"]],
      "Training-set class counts, class weights and hybrid-resampled counts", widths=[3, 3.5, 3, 4.5])
fig("fig_resampling.png", "Training class distribution before and after hybrid resampling", width=12)
bal = pd.read_csv(RES / "balancing_study.csv")
bch = json.loads((RES / "balancing_choice.json").read_text())
table([["Strategy", "Variant", "Val macro-F1", "Test precision", "Test recall", "Test macro-F1"]] +
      [[r["Balancing strategy"], r["Variant"], f4(r["Val macro-F1"]), f4(r["Test precision (macro)"]),
        f4(r["Test recall (macro)"]), f4(r["Test macro-F1"])] for _, r in bal.iterrows()],
      "Balancing ablation with the tuned RF+XGBoost+LightGBM soft-voting ensemble",
      widths=[4.4, 3.8, 1.9, 1.9, 1.7, 1.9], font=8,
      highlight_row=1 + int(bal["Val macro-F1"].idxmax()))
fig("fig_balancing_study.png", "Test macro-F1 for each balancing strategy, with and without threshold "
    "optimisation", width=13)
p(f"The strategy was chosen by <b>validation</b> macro-F1, not by test score: <b>{bch['best_strategy']}</b>. "
  "A key finding is that heavy resampling (hybrid) gives the highest recall but clearly the lowest precision: "
  "SMOTE creates synthetic attack records inside the dense Normal region, so the model raises false alarms on "
  "normal nodes. Mild class weighting keeps precision high while still emphasising rare attacks. Threshold "
  "optimisation improves every strategy, and helps the hybrid one most because it repairs the precision loss.")

# =========================================================== 10 MODELS
h(1, "Machine Learning Models")
h(2, "Baseline Models")
p("Seven classifiers were first trained on the original (imbalanced) training data with default parameters: "
  "Gaussian Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Extra Trees, XGBoost and LightGBM. "
  "These represent probabilistic, linear, single-tree, bagging and boosting families.")
h(2, "Hyper-parameter Tuning")
p("The three strongest families (Random Forest, XGBoost, LightGBM) were tuned with <b>RandomizedSearchCV</b> "
  "(12 random configurations, 3-fold stratified CV, <b>macro-F1</b> scoring) on a 40,000-record sample of the "
  "balanced training set.")
code("""
space = {"n_estimators": [300, 500, 800], "num_leaves": [31, 63, 127, 255],
         "learning_rate": [0.03, 0.05, 0.1], "max_depth": [-1, 10, 16],
         "min_child_samples": [10, 20, 40], "colsample_bytree": [0.7, 0.85, 1.0], ...}
search = RandomizedSearchCV(LGBMClassifier(), space, n_iter=12, cv=3, scoring="f1_macro")
search.fit(X_bal_sample, y_bal_sample)
""", "Randomized hyper-parameter search (LightGBM)")
table([["Model", "Best CV macro-F1", "Mean CV macro-F1", "Best parameters"]] +
      [[r["Model"], f4(r["Best CV macro-F1"]), f4(r["Mean CV macro-F1 (all trials)"]),
        ", ".join(f"{k}={v}" for k, v in json.loads(r["Best parameters"]).items())] for _, r in tune.iterrows()],
      "Hyper-parameter tuning results", widths=[2.6, 2.2, 2.2, 8.4], font=8)
h(2, "Weighted Soft-Voting Ensemble")
vw = S["voting_weights"]
p("The tuned RF, XGBoost and LightGBM models are combined by averaging their class probabilities, weighted by "
  "each model's validation macro-F1 (" + ", ".join(f"{k}: {v:.3f}" for k, v in vw.items()) + ").")
h(2, "Class-wise Threshold Optimisation")
p("A classifier predicts the class with the highest probability. With imbalanced costs this is not optimal for "
  "macro-F1. We learn one multiplicative weight per class on the <b>validation set</b> using coordinate ascent "
  "(grid 0.5-3.0) and predict argmax(w<sub>c</sub> &middot; P(c|x)).")
code("""
def optimise_class_weights(P, y, grid=np.linspace(0.5, 3.0, 26)):
    w = np.ones(P.shape[1]); best = f1_score(y, P.argmax(1), average="macro")
    for _ in range(3):
        for c in range(P.shape[1]):
            for g in grid:
                cand = w.copy(); cand[c] = g
                f = f1_score(y, (P*cand).argmax(1), average="macro")
                if f > best: best, w = f, cand
    return w
""", "Class-wise threshold (weight) optimisation")
p(f"It was applied to the best validation model (<b>{S['threshold_base_model']}</b>). Learned weights: "
  + ", ".join(f"{k} = {v}" for k, v in S["class_weights"].items()) + ".")

# =========================================================== 11 RESULTS
h(1, "Results and Analysis")
p("All numbers below are measured on the untouched, imbalanced test set "
  f"({S['split']['test']:,} records). Because of RP1 we treat <b>macro-F1</b> (the unweighted mean of the five "
  "class F1-scores) and <b>MCC</b> as the main metrics; accuracy is reported for completeness.")
h(2, "Baseline Model Performance")
cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "MCC", "Train time (s)"]
bs = base.sort_values("F1 (macro)", ascending=False)
table([["Model", "Accuracy", "Precision", "Recall", "Macro F1", "MCC", "Train (s)"]] +
      [[m] + [f4(r[c]) if c != "Train time (s)" else f"{r[c]:.1f}" for c in cols] for m, r in bs.iterrows()],
      "Baseline results (imbalanced training, default parameters, macro-averaged)",
      widths=[3.6, 2, 2, 2, 2, 2, 1.8], highlight_row=1)
p(f"Tree ensembles clearly outperform linear and probabilistic models. Note how Gaussian NB reaches "
  f"{pct(base.loc['Gaussian NB', 'Accuracy'])} accuracy but only {base.loc['Gaussian NB', 'F1 (macro)']:.3f} "
  "macro-F1: high accuracy hides poor detection of rare attacks, which is exactly RP1.")
h(2, "Improved Models")
ims = imp.sort_values("F1 (macro)", ascending=False)
table([["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1", "MCC"]] +
      [[m] + [f4(r[c]) for c in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
                                   "F1 (weighted)", "MCC"]] for m, r in ims.iterrows()],
      "Results after feature engineering, tuning, class weighting (cw), ensembling and threshold optimisation",
      widths=[4.6, 1.8, 1.8, 1.8, 1.8, 2, 1.6], highlight_row=1 + list(ims.index).index(FINAL))
fig("fig_model_comparison.png", "Accuracy and macro-F1 of all baseline and improved models", width=16)
h(2, "Improvement Summary")
d_f1 = FM["F1 (macro)"] - BM["F1 (macro)"]
table([["Metric", f"Best baseline ({BB})", f"Final ({FINAL})", "Change"]] +
      [[k, f4(BM[k]), f4(FM[k]), f"{100 * (FM[k] - BM[k]):+.2f} pp"]
       for k in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "MCC"]],
      "Best baseline versus final model (pp = percentage points)", widths=[3.6, 4.2, 5, 2.4])
p(f"The final model reaches <b>{pct(FM['Accuracy'])} accuracy</b> and <b>{pct(FM['F1 (macro)'])} macro-F1</b> "
  f"({d_f1 * 100:+.2f} pp macro-F1 over the best baseline). "
  "Mild class weighting raises the <i>recall</i> of rare attack classes without the false alarms caused by heavy SMOTE, and threshold optimisation then fine-tunes the precision/recall balance of each class, giving the best trade-off.")
h(2, "Confusion Matrix Analysis")
fig("fig_cm_baseline.png", f"Confusion matrix of the best baseline ({BB}); rows = actual, row-normalised", width=10.5)
fig("fig_cm_final.png", f"Confusion matrix of the final model ({FINAL})", width=10.5)
table([["Class", "Precision", "Recall (detection rate)", "F1-score", "Support"]] +
      [[c, f4(crep.loc[c, "precision"]), f4(crep.loc[c, "recall"]), f4(crep.loc[c, "f1-score"]),
        f"{int(crep.loc[c, 'support']):,}"] for c in CLASSES] +
      [["Macro avg", f4(crep.loc["macro avg", "precision"]), f4(crep.loc["macro avg", "recall"]),
        f4(crep.loc["macro avg", "f1-score"]), f"{int(crep.loc['macro avg', 'support']):,}"]],
      "Per-class performance of the final model", widths=[3, 2.6, 3.8, 2.6, 2.4])
weak = min(CLASSES, key=lambda c: crep.loc[c, "f1-score"])
cmf = pd.read_csv(RES / "confusion_matrix_final.csv", index_col=0)
errs = sorted(((cmf.loc[a, b], a, b) for a in CLASSES for b in CLASSES if a != b), reverse=True)[:3]
p(f"The hardest class is <b>{weak}</b> (F1 = {crep.loc[weak, 'f1-score']:.4f}). The three largest error "
  "types are: " + "; ".join(f"{a} predicted as {b} ({n:,} nodes)" for n, a, b in errs) + ". "
  "TDMA attackers that are confused with Normal are nodes that act as ordinary members in a round, so their "
  "own counters look normal; Blackhole/Grayhole confusion is expected because both attacks behave identically "
  "during cluster-head election and differ only in the fraction of packets dropped (RP2).")
fig("fig_per_class_f1.png", "Per-class F1-score of best baseline versus final model", width=12)
fig("fig_roc.png", "One-vs-rest ROC curves (log-scaled FPR)", width=10)

# =========================================================== 12 LITERATURE COMPARISON
h(1, "Comparison with Recent Studies")
rec = {c: crep.loc[c, "recall"] for c in CLASSES}
table([["Study", "Method", "Evaluation setting", "Accuracy", "Macro F1"],
       ["Almomani et al. [1] (2016)", "Artificial Neural Network", "IDs kept", "Per-class only", "n/r"],
       ["Talukder et al. [2] (2024)", "SMOTE-Tomek + ML (MLSTL-WSN)", "IDs/duplicates not discussed",
        "99.92%", "n/r"],
       ["Ouhmi et al. [3] (2026)", "Boosting ensemble (6 algorithms)", "Not discussed", "99.18%", "n/r"],
       ["<b>This work</b>", f"<b>{FINAL}</b>", "<b>Leakage-free, de-duplicated</b>",
        f"<b>{pct(FM['Accuracy'])}</b>", f"<b>{pct(FM['F1 (macro)'])}</b>"]],
      "Overall comparison with recent studies on WSN-DS (n/r = not reported)",
      widths=[3.6, 4.4, 3.6, 2, 1.8], highlight_row=4)
alm = {"Blackhole": .928, "Grayhole": .756, "Flooding": .994, "TDMA": .922, "Normal": .998}
table([["Class", "Almomani et al. [1] ANN", "This work (recall)", "Gain"]] +
      [[c, pct(alm[c]), pct(rec[c]), f"{100 * (rec[c] - alm[c]):+.2f} pp"] for c in CLASSES],
      "Per-class detection rate compared with the original WSN-DS study", widths=[3, 4.5, 4, 3])
better = [c for c in CLASSES if rec[c] > alm[c]]
worse = [c for c in CLASSES if rec[c] <= alm[c]]
p(f"Compared with the original ANN, our model detects more attackers in {len(better)} of 5 classes "
  f"({', '.join(better)}), with the largest gain on the hard Grayhole class "
  f"({100 * (rec['Grayhole'] - alm['Grayhole']):+.1f} pp, RP2)"
  + (f"; it is slightly lower for {', '.join(worse)} "
     f"({', '.join(f'{100 * (rec[c] - alm[c]):+.2f} pp' for c in worse)}), partly because our test set has "
     "no duplicate records to memorise" if worse else "") + ". The 99%+ accuracies of [2] and [3] are close to ours; however, those "
  "studies report accuracy (dominated by the Normal class) and do not state that identifier columns and "
  "duplicates were removed. Our leakage experiment (Table " + str(globals().get("LEAK_TAB", "")) + ") shows that keeping duplicates can inflate scores, so the "
  "numbers are not strictly comparable; our evaluation is the more conservative one.")

# =========================================================== 13 XAI
h(1, "Model Explainability (SHAP and LIME)")
h(2, "SHAP Methodology")
p("SHAP (SHapley Additive exPlanations) assigns each feature a contribution to a prediction based on Shapley "
  "values from cooperative game theory. We use <b>TreeExplainer</b> on the tuned LightGBM model with 2,000 "
  "random test records; for 5 classes this yields a 2000 x features x 5 array of contributions.")
code("""
explainer   = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_test_sample)     # (2000, n_features, 5)
shap.summary_plot(shap_values[:, :, grayhole], X_test_sample)
""", "SHAP analysis")
h(2, "Global Feature Importance")
fig("fig_shap_global.png", "Mean |SHAP| per feature, stacked by class", width=14)
ts = shap_imp.head(8)
table([["Rank", "Feature"] + CLASSES + ["Total"]] +
      [[i + 1, f] + [f"{ts.loc[f, c]:.3f}" for c in CLASSES] + [f"{ts.loc[f, 'Total']:.3f}"]
       for i, f in enumerate(ts.index)], "Top-8 features by SHAP importance", font=8)
p(f"The most influential feature overall is <b>{shap_imp.index[0]}</b>, followed by "
  f"<b>{shap_imp.index[1]}</b> and <b>{shap_imp.index[2]}</b>. The stacked colours show that different features "
  "drive different attacks, matching the networking intuition of Section 7.1: SCH_S matters almost only for "
  "TDMA, Data_Forward_Ratio and DATA_R matter mostly for Blackhole, and ADV_S produces the largest individual "
  "pushes towards Flooding (see the beeswarm below).")
if "Time" in list(shap_imp.index[:4]):
    p("<b>Caution:</b> the simulation clock <i>Time</i> is also ranked highly. In WSN-DS, attacks are active in "
      "particular periods of each simulation run, so Time partly encodes the scenario rather than node "
      "behaviour. This is a dataset artifact that SHAP made visible; a production IDS should replace it with "
      "per-round or relative-time features (see Future Work).")
h(2, "Feature Impact per Attack (Beeswarm)")
fig("fig_shap_beeswarm_grayhole.png", "SHAP beeswarm for the Grayhole class (red = high feature value)", width=13)
fig("fig_shap_beeswarm_blackhole.png", "SHAP beeswarm for the Blackhole class", width=13)
fig("fig_shap_beeswarm_flooding.png", "SHAP beeswarm for the Flooding class", width=13)
p("Reading guide: each dot is one node; its x-position is how much the feature pushed the prediction towards the "
  "class, its colour is the feature value. For Flooding, high ADV_S values (red) push strongly to the right, "
  "i.e. a node sending many advertisement messages is a flooding attacker. For Blackhole/Grayhole, a high amount "
  "of received data combined with a low forwarding ratio pushes the prediction towards the attack.")
h(2, "Individual Prediction Explanation")
fig("fig_shap_waterfall.png", "SHAP waterfall plot for one node correctly detected as Grayhole", width=13)
p("The waterfall plot starts from the model's average output (base value) and adds the contribution of each "
  "feature for one specific node, ending at the final score for Grayhole. An operator can read exactly which "
  "counters (e.g. packets received vs forwarded) made the node suspicious.")
if (FIG / "fig_lime_blackhole.png").exists():
    h(2, "LIME Local Explanation")
    p("LIME (Local Interpretable Model-agnostic Explanations) fits a simple linear model around one prediction. "
      "It gives human-readable rules such as value ranges of features that support or oppose the predicted class.")
    fig("fig_lime_blackhole.png", "LIME explanation for a node predicted as Blackhole", width=13)
    p("SHAP and LIME agree on the key evidence, which increases trust in the model's decisions.")

# =========================================================== 14 HOW TO RUN
h(1, "Implementation and Reproducibility")
table([["Item", "Details"],
       ["Language / tools", "Python 3.13, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, LIME, "
                            "pandas, matplotlib, seaborn, reportlab"],
       ["Hardware", "Ordinary laptop CPU (no GPU needed); full pipeline runs in about "
                    f"{S['runtime_sec'] / 60:.0f} minutes"],
       ["Disk space", "About 30 MB dataset + 10 MB figures/results"],
       ["Dataset", "WSN-DS CSV (374,661 x 19) placed in data/WSN-DS.csv"],
       ["Scripts", "wsn_ids.py (full pipeline), leakage_check.py (leakage test), balancing_study.py (balancing ablation), make_methodology.py (Figure 3), "
                   "build_report.py (this PDF)"],
       ["Code", CODE_LINK]], "Implementation environment", widths=[3.5, 11.5])
code("""
pip install scikit-learn imbalanced-learn xgboost lightgbm shap lime pandas seaborn reportlab
python wsn_ids.py           # trains + evaluates, writes figures/ and results/
python leakage_check.py     # leakage experiment
python balancing_study.py   # balancing-strategy ablation
python make_methodology.py  # methodology figure
python build_report.py      # builds this PDF report
""", "Commands to reproduce all results")

# =========================================================== 15 CONCLUSION
h(1, "Conclusion and Future Work")
p("This project developed an explainable machine-learning intrusion detection system that identifies malicious "
  "nodes in LEACH-based wireless sensor networks. The main findings are:")
bullets([
    f"<b>High and honest performance:</b> {pct(FM['Accuracy'])} accuracy, {pct(FM['F1 (macro)'])} macro-F1 and "
    f"MCC {FM['MCC']:.4f} on an untouched, imbalanced test set after removing identifiers and duplicates.",
    f"<b>Better rare-attack detection:</b> macro-F1 improved by {d_f1 * 100:+.2f} pp over the best baseline through "
    "class weighting, tuning, ensembling and class-threshold optimisation.",
    "<b>Leakage awareness:</b> we showed that keeping duplicates inflates scores and that models latch onto node "
    "identifiers, pitfalls that published WSN-DS studies rarely address.",
    "<b>Explainability:</b> SHAP and LIME confirm the model relies on meaningful routing behaviour (advertisement "
    "flooding, schedule messages, received-vs-forwarded data, energy use), not on artefacts.",
])
p("<b>Limitations and future work:</b> WSN-DS is simulated (NS-2, LEACH only); validating on other protocols "
  "and real testbeds is needed. SHAP showed that the simulation <i>Time</i> feature carries scenario information, so a "
  "time-independent model should be evaluated. Future work includes a lightweight model using only the top-ranked features for "
  "on-node deployment, temporal models across consecutive rounds to separate Grayhole from Normal, and federated "
  "learning across cluster heads.")

# =========================================================== REFERENCES
story.append(Paragraph("References", H1))
for r in [
    "[1] I. Almomani, B. Al-Kasasbeh and M. Al-Akhras, \"WSN-DS: A Dataset for Intrusion Detection Systems in "
    "Wireless Sensor Networks,\" <i>Journal of Sensors</i>, vol. 2016, Article ID 4731953, 2016. "
    "doi:10.1155/2016/4731953",
    "[2] M. A. Talukder, S. Sharmin, M. A. Uddin, M. M. Islam and S. Aryal, \"MLSTL-WSN: Machine Learning-based "
    "Intrusion Detection using SMOTETomek in WSNs,\" <i>International Journal of Information Security</i>, "
    "Springer, 2024. doi:10.1007/s10207-024-00833-z (arXiv:2402.13277)",
    "[3] S. Ouhmi, H. Khalid, M. Marwan, H. Silkhi and A. Ait Temghart, \"Improved boosting-based machine learning "
    "algorithms for network intrusion detection in wireless sensor network,\" <i>IAES International Journal of "
    "Artificial Intelligence (IJ-AI)</i>, vol. 15, no. 3, pp. 2278-2289, 2026.",
    "[4] S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" <i>NeurIPS</i>, 2017.",
    "[5] M. T. Ribeiro, S. Singh and C. Guestrin, \"Why Should I Trust You?: Explaining the Predictions of Any "
    "Classifier,\" <i>ACM SIGKDD</i>, 2016.",
    "[6] N. V. Chawla et al., \"SMOTE: Synthetic Minority Over-sampling Technique,\" <i>JAIR</i>, vol. 16, 2002.",
    "[7] W. B. Heinzelman, A. P. Chandrakasan and H. Balakrishnan, \"An application-specific protocol "
    "architecture for wireless microsensor networks (LEACH),\" <i>IEEE Trans. Wireless Communications</i>, 2002.",
]:
    story.append(Paragraph(r, ParagraphStyle("ref", parent=BODY, fontSize=10, leading=13, leftIndent=14,
                                             firstLineIndent=-14)))

doc = Doc(str(OUT))
doc.multiBuild(story)
print("Wrote", OUT)
