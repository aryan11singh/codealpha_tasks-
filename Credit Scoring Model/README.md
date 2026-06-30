# 💳 Credit Scoring Model — CodeAlpha Internship Task 1

Predicts an individual's **creditworthiness** (loan repayment likelihood) using past financial data and machine learning classification algorithms.

---

## 📌 Task Overview
| Item | Detail |
|------|--------|
| Internship | CodeAlpha — Machine Learning |
| Task | Task 1: Credit Scoring Model |
| Author | Shashank Singh |
| GitHub | [@Shashank17singh](https://github.com/Shashank17singh) |

---

## 🧠 Models Used
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 74.5% | 0.816 |
| Decision Tree | 82.5% | 0.894 |
| Random Forest | 88.0% | 0.963 |
| **Gradient Boosting** ✅ | **89.5%** | **0.970** |

**Best Model: Gradient Boosting** with ROC-AUC of **0.97**

---

## 📊 Features Used
- Age, Annual Income, Loan Amount, Loan Tenure
- Existing Debts, Missed Payments, Employment Years
- Credit History, Loan Purpose, Debt-to-Income Ratio
- Engineered: Income Per Debt, Payment Reliability, Loan-Income Ratio

---

## 🔧 Tech Stack
- **Python 3**
- **Scikit-learn** — ML models, preprocessing, evaluation
- **Imbalanced-learn** — SMOTE for class balancing
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Shashank17singh/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel

# Install dependencies
pip install -r requirements.txt

# Run the model
python credit_scoring_model.py
```

---

## 📁 Project Structure
```
CodeAlpha_CreditScoringModel/
│
├── credit_scoring_model.py   # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Project docs
└── outputs/
    ├── eda_plots.png         # EDA visualizations
    ├── model_evaluation.png  # ROC curves & confusion matrix
    └── feature_importance.png# Feature importance chart
```

---

## 📈 Output Plots
- **EDA Plots** — Target distribution, income analysis, correlation heatmap
- **Model Evaluation** — Confusion matrix, ROC curves, metrics comparison
- **Feature Importance** — Top features driving creditworthiness

---

## 📜 License
MIT License
