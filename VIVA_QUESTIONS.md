# Viva Preparation: Explainable Malicious Node Detection in WSN

CSE-312 Computer Networks (Sessional), CUET.
Team: Al Rubayet (2204002), Md. Rony Hossain (2204005), Imteajul Islam (2204015), Tahsin Ahmed Chowdhury (2204028)

> **Tip:** memorise the key numbers in the table at the end. Short, confident answers work best.
> If you don't know something, say what you *do* know and how you would find the rest.

---

## Part 1: Project Introduction

**Q1. What is your project about, in one or two sentences?**
We built a machine-learning Intrusion Detection System (IDS) for Wireless Sensor Networks. It looks at the
behaviour of each sensor node (messages sent and received, packets forwarded, energy used) and decides
whether the node is normal or is performing one of four Denial-of-Service attacks: Blackhole, Grayhole,
Flooding or TDMA scheduling.

**Q2. Why did you choose this topic?**
Our teacher gave two topics: image-based malware detection and malicious node prediction. We picked
malicious node prediction because it is a networking problem (attacks on a routing protocol), which fits a
Computer Networks course. It also uses tabular data, so it trains in minutes on a normal laptop CPU without a GPU.

**Q3. What problem does it solve in the real world?**
WSNs are used in smart farming, health monitoring, industrial IoT and military surveillance. Nodes are cheap
and unattended, so an attacker can capture one and make it drop or disrupt traffic. Checking thousands of
nodes manually is impossible. Our system flags malicious nodes automatically and explains *why* each one was flagged.

**Q4. What is the input and output of your system?**
- **Input:** 19 numeric features describing one node in one round: whether it is a cluster head, ADV/JOIN/SCH
  messages sent and received, data packets sent and received, packets forwarded to the base station,
  distances, energy, plus our 3 engineered features.
- **Output:** one of 5 classes: Normal, Blackhole, Grayhole, Flooding, TDMA.

**Q5. What are your research problems?**
1. **RP1, class imbalance:** 90.8% of records are Normal and only 0.88% are Flooding. A model that always says
   "Normal" already gets 90.8% accuracy, so accuracy alone is misleading.
2. **RP2, Blackhole vs Grayhole:** the two attacks behave the same way during cluster-head election. The original
   paper detected only 75.6% of Grayhole nodes.
3. **RP3, data leakage:** many papers keep the node-ID columns and duplicate rows, which can inflate scores.
4. **RP4, black-box models:** network operators need explanations, not just predictions.

**Q6. What are your main contributions?**
A leakage-free pipeline (we removed the ID columns and duplicate rows), 3 routing-behaviour features,
ensemble feature selection, a comparison of 4 class-balancing strategies, tuned models, per-class threshold
optimisation, and SHAP and LIME explanations. We also ran an experiment showing how duplicates and IDs affect results.

**Q7. Which algorithms did you use?**
- **Baselines:** Gaussian Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM.
- **Improved:** tuned Random Forest, XGBoost and LightGBM with class weights, a weighted soft-voting ensemble, and
  per-class threshold optimisation.
- **Final model:** tuned LightGBM with class weights and threshold optimisation.
- **Other techniques:** SMOTE and random under-sampling (compared as balancing options), mutual information and
  chi-square (feature selection), RandomizedSearchCV (tuning), SHAP and LIME (explainability).

**Q8. What tools and libraries did you use?**
Python 3.13, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, LightGBM, SHAP, LIME,
matplotlib and seaborn (plots), and ReportLab (to generate the PDF report). Everything runs on a laptop CPU.
The full pipeline takes about 13 minutes, mostly hyper-parameter tuning.

**Q9. What were your final results?**
Accuracy **99.70%**, macro-F1 **97.91%**, MCC **0.9797** on an untouched test set of 70,283 records.
The best baseline (Random Forest) had a macro-F1 of 97.45%, so we improved it by **+0.46 percentage points**.

**Q10. Briefly, how does the pipeline work? (Explain the methodology figure)**
Load WSN-DS → remove ID columns and duplicates → add 3 features → stratified split (72% train, 8% validation,
20% test) → rank features → handle imbalance → train and tune models → ensemble and threshold optimisation →
evaluate on the test set → explain with SHAP and LIME → output the node's class.

---

## Part 2: Dataset

**Q11. Which dataset did you use and where does it come from?**
**WSN-DS** (Almomani et al., *Journal of Sensors*, 2016). It was generated with the **NS-2 network simulator**
running the **LEACH** routing protocol. It has 374,661 records and 19 columns (18 features + 1 label).
It is public, for example on Kaggle.

**Q12. What does one row represent?**
The state of one sensor node in one LEACH round: how many messages it sent and received, how many data
packets it handled, its distances and the energy it used.

**Q13. What is the class distribution?**
Normal 340,066 (90.77%), Grayhole 14,596 (3.90%), Blackhole 10,049 (2.68%), TDMA 6,638 (1.77%),
Flooding 3,312 (0.88%).

**Q14. Were there missing values or duplicates?**
There were no missing values. There were 8,873 exact duplicate rows, and 23,250 duplicates once the ID
columns were dropped. After cleaning, 351,411 unique records remain.

**Q15. Name a few important features and what they mean.**
- `Is_CH`: whether the node is a cluster head.
- `ADV_S`: advertisement messages sent. This is high for Flooding.
- `SCH_S`: TDMA schedule messages sent. This is high for the TDMA attack.
- `DATA_R`: data packets received by a cluster head.
- `Data_Sent_To_BS`: packets forwarded to the base station. This is low or zero for Blackhole.
- `Expanded Energy`: energy used in the round.

---

## Part 3: Networking Concepts

**Q16. What is a Wireless Sensor Network?**
A network of small, battery-powered sensor nodes that sense the environment and send data wirelessly over
multiple hops to a **base station (sink)**. The main constraints are energy, memory, processing power and bandwidth.

**Q17. What is LEACH?**
Low-Energy Adaptive Clustering Hierarchy, a hierarchical routing protocol. Each round has two phases:
- **Setup phase:** some nodes randomly elect themselves cluster heads (CH) and broadcast ADV messages. Other nodes
  send JOIN requests to the nearest CH. The CH then sends a TDMA schedule (SCH) giving each member a time slot.
- **Steady-state phase:** members send data in their slot. The CH aggregates it and forwards it to the base station.

The CH role rotates between nodes so that energy is spent evenly.

**Q18. Explain each attack.**
- **Blackhole:** the attacker advertises itself as a CH, attracts its neighbours' traffic, then drops **all** packets.
- **Grayhole (selective forwarding):** the same trick, but it drops packets **selectively or randomly**, so it is
  harder to detect.
- **Flooding:** the attacker sends a huge number of ADV messages, which wastes energy and bandwidth and disrupts
  CH selection.
- **Scheduling (TDMA) attack:** a malicious CH gives every member the **same time slot**, so packets collide and are lost.

**Q19. Which network layer do these attacks target?**
Mainly the **network/routing layer** (Blackhole, Grayhole, Flooding) and the **MAC layer** (the TDMA scheduling
attack). All of them are Denial-of-Service attacks.

**Q20. Why is Grayhole harder to detect than Blackhole?**
A Blackhole forwards nothing, so its forwarding ratio is about 0, which is very clear. A Grayhole forwards
*some* packets, so in some rounds it looks like a normal CH that just had packet loss.

**Q21. What is an IDS? Signature-based vs anomaly-based?**
An Intrusion Detection System monitors activity and raises alerts.
- **Signature-based** IDSs match known attack patterns. They are fast, but they miss new attacks.
- **Anomaly/ML-based** IDSs learn normal and attack behaviour from data. They can generalise, but they may
  produce false alarms.

Ours is ML-based and supervised.

**Q22. Where would this IDS run in a real network?**
Most realistically at the base station or the cluster heads, which have more resources. Nodes report their
counters and the model classifies them. A lightweight version using only the top features could run on the
cluster heads themselves.

---

## Part 4: Preprocessing and Feature Engineering

**Q23. Why did you remove the `id` and `who_CH` columns?**
They are node identifiers. In the simulation, the same node IDs are always the attackers, so a model could
memorise *which* node is bad instead of learning *how* a bad node behaves. In a real network new nodes have
new IDs, so that knowledge would be useless. In our experiment, LightGBM ranked the ID columns 2nd and 3rd
in importance when they were kept.

**Q24. Why remove duplicates *before* splitting?**
If the same row appears in both the training and test sets, the model has effectively seen the test data,
so the score is inflated. In our experiment, keeping duplicates raised Random Forest's macro-F1 by about 0.7 points.

**Q25. What features did you engineer, and why?**
- `Data_Forward_Ratio = Data_Sent_To_BS / (DATA_R + 1)` measures whether a CH forwards what it receives.
  It is about 0 for Blackhole and low for Grayhole.
- `Ctrl_Msgs_Sent = ADV_S + JOIN_S + SCH_S` counts control-message activity, which is high for Flooding and TDMA.
- `Energy_per_Packet = Energy / (DATA_S + DATA_R + 1)` captures energy wasted per useful packet.

The `+1` avoids division by zero.

**Q26. Why stratified splitting?**
It keeps the same class proportions in the train, validation and test sets. Without it, the rare Flooding
class might be badly under-represented in one of the sets.

**Q27. Why did you need a validation set as well as a test set?**
We tuned decisions (ensemble weights, class thresholds, which balancing strategy to use) on the validation set.
If we had tuned them on the test set, the test score would be optimistic. The test set is used only once, at the end.

---

## Part 5: Feature Selection

**Q28. How did you select features?**
We combined three scores, computed on training data only and normalised to 0–1: **0.4 × mutual information +
0.2 × chi-square + 0.4 × Random Forest importance**. We then tested the top k features (k = 4 to 19) with
3-fold cross-validated macro-F1.

**Q29. What are mutual information and chi-square?**
- **Mutual information** measures how much knowing a feature reduces uncertainty about the class. It also
  captures non-linear relationships.
- **Chi-square** is a statistical test of dependence between a feature and the class. It needs non-negative
  values, so we min-max scaled the features first.

**Q30. How many features did you finally keep, and why?**
All 19. Every reduction lowered macro-F1: the top 12 gave 0.955 against 0.967 for all 19. The ranking is still
useful for a lightweight deployment. The top features were ADV_S, Is_CH, Ctrl_Msgs_Sent, Energy_per_Packet and
Expanded_Energy.

---

## Part 6: Class Imbalance

**Q31. Why is class imbalance a problem?**
The model can reach high accuracy by mostly predicting the majority class (Normal) and ignoring rare attacks,
which are exactly the ones we care about.

**Q32. What is SMOTE?**
Synthetic Minority Over-sampling Technique. For a minority sample, it picks one of its k nearest neighbours of
the same class and creates a new synthetic point on the line between them. We applied it to training data only.

**Q33. Which balancing strategies did you compare, and which won?**
1. No resampling
2. Square-root inverse class weights
3. Moderate SMOTE (attack classes up to 15,000 records)
4. Hybrid: under-sample Normal to 60,000 and SMOTE each attack class to 20,000

Based on validation macro-F1, **class weights** won.

**Q34. Interesting finding: why did heavy SMOTE perform worst?**
It gave the highest recall but the lowest precision. SMOTE created synthetic attack points inside the dense
Normal region, so the model started raising false alarms on normal nodes. Mild class weights emphasise rare
classes without creating fake data.

**Q35. What are class weights?**
Each training record of class c gets the weight sqrt(N_max / N_c), so mistakes on rare classes cost more in the
loss function. We used the square root so the weights aren't extreme. The largest weight was Flooding at 10.13.

---

## Part 7: Algorithms

**Q36. How does a Decision Tree work?**
It splits the data step by step on feature thresholds that best separate the classes (using Gini impurity or
entropy) until the leaves are mostly one class. It is easy to interpret but overfits easily.

**Q37. How does Random Forest work?**
It is an ensemble of many decision trees, each trained on a bootstrap sample with a random subset of features
at each split. The prediction is a majority vote. This is called **bagging**, and it reduces variance and overfitting.

**Q38. What is Extra Trees?**
Extremely Randomised Trees. It is like Random Forest, but split thresholds are chosen at random, which makes
it faster and adds more randomness.

**Q39. What is boosting? How do XGBoost and LightGBM work?**
Boosting builds trees **sequentially**, and each new tree corrects the errors (gradients) of the previous ones.
- **XGBoost:** gradient boosting with regularisation, built-in handling of sparse data, and level-wise tree growth.
- **LightGBM:** gradient boosting with **histogram-based** splits and **leaf-wise** growth (it always splits the
  leaf with the largest loss reduction). It is very fast and accurate on large tabular data.

**Q40. Why did LightGBM perform best?**
It handles 250k+ rows quickly, its leaf-wise growth captures complex patterns, and its tuned settings
(num_leaves=127, 800 trees, learning rate 0.1) fit the data well. Combined with class weights and thresholds,
it gave the best validation macro-F1.

**Q41. Why did Naive Bayes perform poorly?**
It assumes the features are independent and Gaussian. Our features are counts that are strongly correlated
and skewed. It reached 95.8% accuracy but only 0.754 macro-F1.

**Q42. What is soft voting?**
It averages the predicted class *probabilities* of several models (here weighted by each model's validation F1)
and picks the class with the highest average. Hard voting just counts the predicted labels.

**Q43. What is hyper-parameter tuning? Why RandomizedSearchCV instead of GridSearchCV?**
Hyper-parameters are settings chosen before training, such as the number of trees, tree depth and learning
rate. RandomizedSearchCV tries a fixed number of random combinations (12 in our case) with 3-fold
cross-validation. Grid search tries every combination, which would take far longer for a similar result.

**Q44. What is class-wise threshold optimisation?**
Normally the model picks the class with the highest probability. We learned one multiplier per class on the
validation set (searching 0.5 to 3.0 with coordinate ascent) to maximise macro-F1, then predict
argmax(weight_c × P(c)). It shifts each class's decision boundary.

**Q45. What is cross-validation?**
The training data is split into k folds. The model trains on k−1 folds and is evaluated on the remaining one,
repeated k times, and the scores are averaged. It gives a more reliable estimate than a single split.

---

## Part 8: Evaluation Metrics

**Q46. Define precision, recall and F1.**
- **Precision** = TP / (TP + FP): of the nodes flagged as an attack, how many really were.
- **Recall (detection rate)** = TP / (TP + FN): of the real attackers, how many we caught.
- **F1** = 2PR / (P + R): the harmonic mean of the two.

**Q47. Why did you use macro-F1 as the main metric instead of accuracy?**
Macro-F1 averages the F1 of all 5 classes **equally**, so the rare Flooding class counts as much as Normal.
Accuracy is dominated by Normal (90%+). Weighted F1 is also dominated by Normal.

**Q48. What is MCC?**
Matthews Correlation Coefficient. It ranges from −1 to +1 and uses all cells of the confusion matrix, so it
is robust under class imbalance. Ours is 0.9797.

**Q49. How do you read a confusion matrix?**
Rows are actual classes and columns are predicted classes. The diagonal shows correct predictions and
off-diagonal cells show mistakes. Our largest error was TDMA predicted as Normal (90 nodes, about 7% of TDMA).

**Q50. Which class was hardest, and why?**
**TDMA** (F1 0.958, recall 92.9%). Some TDMA records are nodes acting as ordinary members in that round, so
their own counters look normal. Flooding had F1 0.960 because its precision was lower (some normal nodes
with bursts of ADV messages were flagged).

**Q51. What is a ROC curve and AUC?**
A ROC curve plots the true positive rate against the false positive rate at different thresholds. AUC is the
area under it: 1.0 is perfect and 0.5 is random. We plotted one-vs-rest ROC curves per class, and all AUCs are
close to 1.

---

## Part 9: Explainability (SHAP and LIME)

**Q52. What is SHAP?**
SHapley Additive exPlanations. It is based on Shapley values from cooperative game theory: each feature is a
"player", and its SHAP value is its fair contribution to moving the prediction away from the average output.
We used TreeExplainer, which computes these values exactly and quickly for tree models.

**Q53. What do your SHAP plots show?**
- **Global bar plot:** average |SHAP| per feature, stacked by class. The top features were Time, ADV_S, ADV_R
  and Energy_per_Packet.
- **Beeswarm:** one dot per node, coloured by feature value. For example, for Flooding, high ADV_S values (red)
  push strongly towards Flooding. For Blackhole, a low Data_Forward_Ratio and high DATA_R push towards Blackhole.
- **Waterfall:** a breakdown for one single node, showing how each feature moved it from the base value to
  "Grayhole".

**Q54. What is LIME, and how is it different from SHAP?**
Local Interpretable Model-agnostic Explanations. It perturbs one sample, watches how the model's output
changes, and fits a simple linear model locally, giving rules like "ADV_S > 0 supports Blackhole". LIME is
local and approximate. SHAP has a solid theoretical basis and works both locally and globally.

**Q55. SHAP shows `Time` as the top feature. Isn't that a problem?**
Yes, and we report it as a limitation. `Time` is the simulation clock. In WSN-DS, attacks run during certain
periods of each simulation, so `Time` partly encodes the scenario rather than node behaviour. A real IDS should
use per-round or relative time features instead. Finding problems like this is exactly why explainability matters.

---

## Part 10: Results and Comparison

**Q56. Compare your results with other papers.**
| Study | Result |
|---|---|
| Almomani et al. 2016 (ANN) | detection rate per class: Blackhole 92.8%, Grayhole 75.6%, Flooding 99.4%, TDMA 92.2%, Normal 99.8% |
| Talukder et al. 2024 (SMOTE-Tomek + ML) | 99.92% multiclass accuracy |
| Ouhmi et al. 2026 (boosting) | 99.18% accuracy |
| **Ours** | **99.70% accuracy, 97.91% macro-F1** |

We beat the original paper in 4 of 5 classes (Grayhole +23.3 points) but are 1.3 points lower on Flooding.

**Q57. Your accuracy is lower than Talukder et al.'s 99.92%. Is your model worse?**
Not necessarily. They report accuracy, which is dominated by the Normal class, and they don't say whether they
removed duplicate rows or ID columns. Our evaluation is stricter: no duplicates and no IDs. We showed that
keeping duplicates inflates scores. So the numbers are not directly comparable.

**Q58. How much did your improvements help?**
Macro-F1 went from 97.45% (best baseline) to 97.91%, precision from 97.51% to 98.06%, and MCC from 0.9756 to 0.9797.

---

## Part 11: Limitations, Future Work and Tricky Questions

**Q59. What are the limitations?**
The data is simulated (NS-2, LEACH only), so it may not match real hardware. The `Time` feature is a dataset
artifact. There is no temporal modelling across rounds. We tested on only one dataset.

**Q60. What would you do next?**
Test on other protocols and real testbeds, build a time-independent model, train a lightweight model with the
top features for cluster heads, model several consecutive rounds to separate Grayhole from Normal, and try
federated learning.

**Q61. Is SMOTE applied to the test set?**
No. Resampling and class weights are applied to the training data only. The test set keeps the real imbalance,
otherwise the evaluation would be unrealistic.

**Q62. What is overfitting, and how did you avoid it?**
Overfitting is when a model memorises the training data and fails on new data. We avoided it by removing
duplicates and IDs, using cross-validation during tuning, using a separate validation set for decisions, and
evaluating once on an untouched test set.

**Q63. Is this supervised or unsupervised learning?**
Supervised: every record has a label (Attack type).

**Q64. How long does training take? Do you need a GPU?**
No GPU is needed. Training the final models takes under a minute each. The whole pipeline, including tuning,
takes about 13 minutes on a laptop CPU.

**Q65. What is the role of AI tools in your project?**
We used an AI coding assistant to help plan the pipeline, write code and draft the report. We ran every
experiment ourselves, checked the results, and can explain each step.

---

## Key Numbers to Remember

| Item | Value |
|---|---|
| Dataset | WSN-DS, 374,661 records, 18 features, 5 classes |
| After cleaning | 351,411 records, 19 features (16 + 3 engineered) |
| Split | 253,015 train / 28,113 validation / 70,283 test |
| Imbalance | Normal 90.8%, Flooding 0.88% (103:1) |
| Best baseline | Random Forest: accuracy 99.64%, macro-F1 97.45% |
| Final model | Tuned LightGBM + class weights + thresholds |
| Final results | Accuracy 99.70%, macro-F1 97.91%, MCC 0.9797 |
| Hardest class | TDMA (F1 0.958) |
| Biggest gain vs 2016 paper | Grayhole detection 75.6% → 98.9% |
