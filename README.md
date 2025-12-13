<p align="center">
  <img src="assets/bcgx-logo.png" alt="BCG X logo" width="500">
</p>

<h1 align="center">PowerCo Churn Prediction (BCG X – Forage)</h1>
<p align="center"><em>BCG X è un marchio dei rispettivi proprietari.</em></p>

## Overview
Obiettivo: stimare quali clienti PowerCo hanno maggiore probabilità di **churn nei 3 mesi successivi** (`churn`) per abilitare una strategia di retention basata su:
1) **ranking top-k** (chi contattare prima)
2) **incentivo selettivo** (solo quando economicamente conveniente)

**Key results (indicativi sul dataset Forage)**
- **Best setup:** Random Forest + feature “bill pressure”
- **PR-AUC (5-fold stratified):** ~0.3215
- **Top-20% targeting:** Precision ~0.2315 | Recall ~0.4761 | Lift ~2.38×

---

## Data
Il progetto utilizza due sorgenti:
- `client_data.csv` (1 riga per cliente, include target `churn`)
- `price_data.csv` (più righe per cliente, per data)

> Dataset non incluso nel repository.

**Data dictionary**
- Vedi `Data Description.pdf` per la descrizione dei campi.

---

## Forage tasks vs this repository
Il corso Forage fornisce dataset diversi nelle varie task (arricchiti progressivamente).
In questo repository ho svolto l’intero workflow **end-to-end partendo esclusivamente dai dataset originali (Task 2)**, ricostruendo da zero le feature (incluse quelle che Forage fornisce già pronte nelle task successive).

---

## Temporal reference (as-of features)
Il dataset non include una colonna esplicita di “snapshot date” rispetto alla quale è calcolata la label `churn`.
Per garantire coerenza con il dataset prezzi (disponibile fino al **31/12/2015**), ho adottato:
- **Reference date: `T = 2015-12-31`**

Le feature temporali sono state costruite come differenze rispetto a `T` (es. tenure e giorni al rinnovo).
Alcune date contrattuali (`date_renewal`, `date_end`) possono essere successive a `T` e sono interpretate come informazioni pianificate/registrate.
Per campi operativi come `date_modif_prod` (che può superare `T`), le feature temporali vanno interpretate con cautela: in uno scenario reale si adotterebbe un approccio conservativo “as-of” per evitare di usare informazione post-`T` come se fosse già osservabile.

---

## Workflow

### 1) Exploratory Data Analysis (EDA)
- Controllo qualità: missingness e duplicati (`id` unico su client; `(id, price_date)` su price).
- Target distribution: churn rate ~9–10% (dataset sbilanciato).
- Analisi distribuzioni numeriche (consumi/margini skewed) → scelta `log1p` in FE.
- Analisi preliminare sui prezzi: segnali su prezzi fissi e differenze peak/off-peak.

### 2) Feature Engineering (FE)
**Client features (as-of T)**
- `tenure_days = (T - date_activ).days`
- `days_to_renewal = (date_renewal - T).days`
- `days_since_prod_modif = (T - date_modif_prod).days`
- `dev_consum = cons_last_month - (cons_12m/12)`

**Price features (from `price_data`)**
Per ciascun cliente ho aggregato le serie prezzi 2015:
- statistiche: `mean`, `std`, `last`
- trend: `slope` (variazione nel tempo)
- differenze/rapporti tariffari (peak vs off-peak)

**Bill pressure (proxy bolletta)**
Feature che combinano prezzi e utilizzo/potenza (es. prezzo fisso × potenza, prezzo variabile × consumo) per catturare “pressione economica” oltre al prezzo nudo.

**Trasformazioni**
- `log1p` su variabili fortemente skewed (consumi, margini, potenza, deviazioni).

> Scelta prudenziale: le colonne `forecast_*` sono state trattate con cautela (potenziale leakage o non replicabilità in assenza di una definizione temporale chiara della label). Estensione naturale: ablation “con vs senza forecast”.

### 3) Modeling
- Preprocessing: imputazione (median/mode) + one-hot encoding per categoriche.
- Modelli: Logistic Regression (baseline) e Random Forest (non-lineare).
- Metriche: **PR-AUC (Average Precision)** + metriche operative top-k.

**Ablation test**
- baseline (feature core)
- no_price (senza feature prezzo)
- bill_pressure (con proxy bolletta)

Takeaway: l’informazione di prezzo migliora soprattutto quando trasformata in **bill pressure**.

---

## Results

### Dataset & target
- Target: `churn` = cliente churn nei **prossimi 3 mesi**
- Churn rate (base rate): **~9.7%** (indicativo sul dataset fornito)

### Model performance (PR-AUC / Average Precision)
Valutazione su classificazione sbilanciata usando **PR-AUC (Average Precision)**.

**Holdout (split singolo) — Random Forest**
| Feature set | PR-AUC (AP) |
|---|---:|
| baseline | ~0.3273 |
| no_price | ~0.3234 |
| bill_pressure | **~0.3326** |

**Cross-validation (5-fold stratified) — confronto feature set**
| Feature set | AP mean |
|---|---:|
| bill_pressure | **~0.3215** |
| baseline | ~0.3169 |
| no_price | ~0.3043 |

### Operational effectiveness (top-k targeting)
Esempio: targeting **Top-20%** (≈ 20% clienti più a rischio)
- Precision@20%: **~0.2315**
- Recall@20%: **~0.4761**
- Lift@20% vs base rate: **~2.38×**

Interpretazione: contattando ~1/5 della base clienti si intercetta ~metà dei churner, con una concentrazione di churn ~2.4× superiore al caso casuale.

### Explainability (permutation importance)
Le feature più rilevanti (Permutation Importance su Random Forest) includono:
- margini: `margin_net_pow_ele`, `margin_gross_pow_ele`
- variabili contrattuali: `days_to_renewal`, `tenure_days`
- variabili di segmentazione: `origin_up`, `channel_sales`
- segnali di “bill pressure”: es. `bill_fix_off_last_x_pow`
- segnali di variazione consumo: `dev_consum`

Nota: la permutation importance può distribuire importanza tra feature correlate; è indicativa, non causale.

---

## Decision layer (model → action)
Il modello viene usato come ranking:
- selezione **top-k** clienti più a rischio (es. top-20%) per massimizzare recall con budget limitato
- incentivo solo se break-even positivo: beneficio atteso ≥ costi (contatto + incentivo), usando `net_margin` come proxy del valore cliente e assunzioni esplicite di uplift

---

## Next improvements
- Calibrazione delle probabilità (Platt/Isotonic) se si usano soglie economiche.
- A/B test per stimare uplift reale di contatto e incentivo.
- Validazione temporale più rigorosa se disponibile una timeline completa per la label.
- Ablation dedicata sulle feature `forecast_*`.
- Modelli boosting (LightGBM/XGBoost) e tuning strutturato.

---

## How to run
```bash
pip install -r requirements.txt
```
### Esegui in ordine:
-	01_eda.ipynb
-	02_feature_engineering_and_modelling.ipynb

> Nota: i dataset non sono inclusi nel repository.

---

## Notes

Progetto svolto come parte della job simulation **BCG X - Data Science - su Forage**.
Codice e analisi sono originali e realizzati da me.
