# Airline Passenger Satisfaction — ML Classification

**Can we predict whether an airline passenger will be satisfied or dissatisfied?**  
This project applies seven machine learning models to a 103,000+ passenger survey dataset to identify the key drivers of satisfaction and build a reliable prediction tool.

---

## Project Overview

| | |
|---|---|
| **Dataset** | 103,904 airline passenger survey responses (Kaggle) |
| **Target variable** | `satisfaction` (Satisfied / Neutral or Dissatisfied) |
| **Class split** | 43.3% satisfied, 56.7% neutral/dissatisfied |
| **Models** | Naive Bayes, Logistic Regression, Decision Tree, XGBoost, Random Forest, ANN, SVM |
| **Best accuracy** | **96.50% (Random Forest)** |
| **Tools** | Python, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels |

---

## Model Performance Summary

| Model | Accuracy | Notes |
|---|---|---|
| **Random Forest** | **96.50%** | Best overall; strong generalisation |
| **XGBoost** | 96.27% | Fast on large data; lower interpretability |
| **ANN** | ~95%+ | Handles non-linear patterns well |
| **SVM** | ~94%+ | Solid; computationally expensive |
| **Decision Tree** | ~88%+ | Interpretable; prone to overfitting |
| **Logistic Regression** | ~87%+ | Simple; good baseline |
| **Naive Bayes** | ~85%+ | Fast; assumes feature independence |

---

## Key Findings

**Top drivers of passenger satisfaction:**

- 🛜 **Inflight Wi-Fi** — single strongest predictor; poor Wi-Fi strongly predicts dissatisfaction
- 🪑 **Seat comfort** — critical especially on long-haul flights
- 🛂 **Online boarding** — frictionless digital experience matters
- ✈️ **Business class passengers** are significantly more satisfied than economy
- 💼 **Business travellers** are more satisfied than personal travellers

**Segment insights (K-Means clustering):**
- Loyal business-class passengers: highest satisfaction
- Disloyal personal-travel economy passengers: lowest satisfaction
- Marginal passengers (ratings 2–4) represent the highest conversion opportunity

**Business recommendations:**
1. Prioritise Wi-Fi upgrades and online boarding improvements — highest ROI on satisfaction
2. Target "marginal" passengers (mid-range satisfaction ratings) with lightweight interventions (loyalty points, targeted surveys)
3. Apply the **Peak-End Rule** — optimise the final moments of the journey (arrival experience, baggage) disproportionately

---

## Notebook Structure

```
1. Introduction & Business Understanding
2. Data Understanding (EDA)
   ├── Target distribution
   ├── Numerical & categorical feature analysis
   ├── Correlation analysis
   └── Train/test split handling
3. Model Applying
   ├── 4.1 Naive Bayes
   ├── 4.2 Logistic Regression + K-Means clustering
   ├── 4.3 Decision Tree + XGBoost
   ├── 4.4 Random Forest (best model)
   ├── 4.5 ANN + feature importance
   └── 4.6 SVM
4. Conclusion & Business Insights
```

---

## How to Run

**1. Download data from Kaggle:**  
[Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) → place `train.csv` and `test.csv` in `data/`

**2. Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow category_encoders statsmodels
```

**3. Run the notebook:**
```bash
jupyter notebook airline_passenger_satisfaction.ipynb
```

---

## Repo Structure

```
airline-passenger-satisfaction/
├── airline_passenger_satisfaction.ipynb   # Main analysis notebook
├── README.md
└── data/
    ├── train.csv    # Download from Kaggle (not included)
    └── test.csv     # Download from Kaggle (not included)
```

---

## Academic Context

Group project for **INFS5720**, UNSW Sydney (2025).  
Contributors: Yunwei Xu, Fan Wu, Xi Lu, Jiashen Zhou, Xinyi Li.  
Data: [Kaggle — TJ Klein](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

---

*Xi Lu (Lucy) — UNSW Master of Commerce (Marketing & Business Analytics), 2026*
