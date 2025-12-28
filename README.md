<p align="center">
  <img src="assets/bcgx-logo.png" alt="BCG X logo" width="500">
</p>

# PowerCo Churn Prediction — Retention Targeting (Python • ML)

**Goal:** predict which PowerCo customers are most likely to **churn within the next 3 months** (`churn`) to support:
1) **top-k ranking** (who to contact first)  
2) **selective incentives** (only when economically justified)

**Key results (BCG X - Forage dataset):**
- **Best setup:** Random Forest + **bill pressure** features  
- **PR-AUC/Average Precision (5-fold stratified): ~0.3215**  
- **Top-20% targeting:** Precision ~0.232 | Recall ~0.476 | **Lift ~2.38×**

> Project completed as part of the **BCG X – Data Science** job simulation on **Forage**. Code and analysis are original.  
> Datasets are not included in the repository. About the simulation: [Forage - BCG X Job Simulation - Data Science](https://www.theforage.com/simulations/bcg/data-science-ccdz)

---

## What I built (end-to-end)
- **EDA → Feature engineering → Modeling → Evaluation → Decision layer**
- **Imbalanced evaluation:** PR-AUC (Average Precision) + **top-k operational metrics**
- **Actionable output:** prioritized customer list (top-k) + incentive rule (break-even)

---

## Data
- `client_data.csv` (1 row per customer, includes target `churn`)
- `price_data.csv` (multiple rows per customer over time)  
Data dictionary: [DataDescription.pdf](DataDescription.pdf)

---

## Time consistency (“as-of” features)
No explicit snapshot date is provided for the churn label.  
To align with price data availability (up to **2015-12-31**), I set:
- **Reference date: `T = 2015-12-31`**

Temporal features are computed as deltas vs `T` (e.g., tenure, days to renewal).  
Some contract dates can be > `T` and are treated as planned/registered information; operational fields are handled cautiously to reduce leakage risk.

---

## Modeling
- Baseline: **Logistic Regression**
- Main model: **Random Forest**
- Preprocessing: missing value imputation + one-hot encoding  
- **Ablation:** baseline vs no_price vs bill_pressure  
**Takeaway:** price information helps most when transformed into **bill pressure**.

---

## Results

### PR-AUC (Average Precision)
**Cross-validation (5-fold stratified)**
| Feature set | AP mean |
|---|---:|
| bill_pressure | **~0.3215** |
| baseline | ~0.3169 |
| no_price | ~0.3043 |

### Top-k targeting (test set)
Targeting **Top-20%** highest-risk customers:
- Precision@20%: **~0.232**
- Recall@20%: **~0.476**
- Lift@20% vs base rate: **~2.38×**

**Lift table**
| Contacted (top%) | Precision | Recall | Lift vs base |
|---:|---:|---:|---:|
| 1%  | 0.861 | 0.087 | 8.86× |
| 2%  | 0.671 | 0.138 | 6.91× |
| 5%  | 0.484 | 0.248 | 4.97× |
| 10% | 0.337 | 0.346 | 3.47× |
| 15% | 0.278 | 0.428 | 2.86× |
| 20% | 0.232 | 0.476 | 2.38× |
| 30% | 0.196 | 0.606 | 2.02× |

---

## Decision layer (model → action)
Use the model as a ranking tool:
- contact customers in **top-k** (e.g., top-10% / top-20%)
- offer incentives only when expected value is positive

**Expected value (general form):**  
`P(churn) × uplift × margin − cost`

---

## How to run
```bash
pip install -r requirements.txt
```
---

### Run notebooks in order:
-	[01_eda.ipynb](01_eda.ipynb)
-	[02_feature_engineering_and_modelling.ipynb](02_feature_engineering_and_modelling.ipynb)

#### *BCG X and Forage are trademarks of their respective owners.*

