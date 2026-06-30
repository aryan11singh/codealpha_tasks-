# =============================================================
# CodeAlpha Internship — Task 1: Credit Scoring Model
# Author: Shashank Singh
# GitHub: https://github.com/Shashank17singh
# =============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import os
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("   CodeAlpha — Credit Scoring Model")
print("=" * 60)

# ── 2. GENERATE REALISTIC SYNTHETIC DATASET ──────────────────
# (Mirrors UCI German Credit / Give Me Some Credit structure)
np.random.seed(42)
n = 1000

age           = np.random.randint(21, 70, n)
income        = np.random.randint(15000, 120000, n)
loan_amount   = np.random.randint(1000, 50000, n)
loan_tenure   = np.random.randint(6, 60, n)          # months
existing_debts= np.random.randint(0, 5, n)
missed_payments = np.random.randint(0, 10, n)
employment_years= np.random.randint(0, 30, n)
credit_history= np.random.choice(['Good', 'Average', 'Poor'], n, p=[0.5, 0.3, 0.2])
loan_purpose  = np.random.choice(['Home', 'Car', 'Education', 'Personal'], n)
debt_to_income= np.round(loan_amount / (income + 1), 4)

# Target: 1 = creditworthy (low risk), 0 = not creditworthy (high risk)
# Logically influenced by key features
credit_score = (
    (income > 50000).astype(int) * 2 +
    (missed_payments < 3).astype(int) * 3 +
    (employment_years > 3).astype(int) * 2 +
    (existing_debts < 2).astype(int) * 1 +
    (credit_history == 'Good').astype(int) * 3 +
    (credit_history == 'Average').astype(int) * 1 +
    (debt_to_income < 0.3).astype(int) * 2
)
# Convert to binary with some noise
threshold = credit_score.mean()
noise = np.random.normal(0, 0.5, n)
target = ((credit_score + noise) >= threshold).astype(int)

df = pd.DataFrame({
    'Age': age,
    'Annual_Income': income,
    'Loan_Amount': loan_amount,
    'Loan_Tenure_Months': loan_tenure,
    'Existing_Debts': existing_debts,
    'Missed_Payments': missed_payments,
    'Employment_Years': employment_years,
    'Credit_History': credit_history,
    'Loan_Purpose': loan_purpose,
    'Debt_To_Income_Ratio': debt_to_income,
    'Creditworthy': target
})

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   Creditworthy (1): {target.sum()}  |  Not Creditworthy (0): {(target==0).sum()}")
print("\nFirst 5 rows:")
print(df.head())

# ── 3. EDA ────────────────────────────────────────────────────
print("\n\n📈 EDA — Exploratory Data Analysis")
print("-" * 40)
print(df.describe())
print(f"\nMissing values:\n{df.isnull().sum()}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Credit Scoring Model — EDA', fontsize=16, fontweight='bold')

# Target distribution
axes[0,0].pie(df['Creditworthy'].value_counts(), labels=['Creditworthy','Not Creditworthy'],
              autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'], startangle=90)
axes[0,0].set_title('Target Distribution')

# Income vs Creditworthy
df.boxplot(column='Annual_Income', by='Creditworthy', ax=axes[0,1],
           patch_artist=True)
axes[0,1].set_title('Income by Creditworthiness')
axes[0,1].set_xlabel('Creditworthy (1=Yes, 0=No)')

# Missed Payments distribution
axes[0,2].hist([df[df['Creditworthy']==1]['Missed_Payments'],
                df[df['Creditworthy']==0]['Missed_Payments']],
               bins=10, label=['Creditworthy','Not Creditworthy'],
               color=['#2ecc71','#e74c3c'], alpha=0.7)
axes[0,2].set_title('Missed Payments Distribution')
axes[0,2].legend()

# Credit History count
ch_counts = df.groupby(['Credit_History','Creditworthy']).size().unstack()
ch_counts.plot(kind='bar', ax=axes[1,0], color=['#e74c3c','#2ecc71'], alpha=0.8)
axes[1,0].set_title('Credit History vs Creditworthy')
axes[1,0].tick_params(axis='x', rotation=0)

# Debt-to-Income ratio
axes[1,1].scatter(df['Debt_To_Income_Ratio'], df['Annual_Income'],
                  c=df['Creditworthy'], cmap='RdYlGn', alpha=0.5, s=20)
axes[1,1].set_title('Debt-to-Income vs Annual Income')
axes[1,1].set_xlabel('Debt-to-Income Ratio')
axes[1,1].set_ylabel('Annual Income')

# Correlation heatmap (numeric only)
num_cols = df.select_dtypes(include=np.number).columns
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[1,2], annot=True, fmt='.2f', cmap='coolwarm',
            annot_kws={'size': 7})
axes[1,2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('outputs/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ EDA plots saved → outputs/eda_plots.png")

# ── 4. PREPROCESSING ─────────────────────────────────────────
print("\n\n🔧 Preprocessing")
print("-" * 40)

# Encode categoricals
le = LabelEncoder()
df['Credit_History_Enc'] = le.fit_transform(df['Credit_History'])
df['Loan_Purpose_Enc']   = le.fit_transform(df['Loan_Purpose'])

# Feature engineering
df['Income_Per_Debt']    = df['Annual_Income'] / (df['Existing_Debts'] + 1)
df['Payment_Reliability']= 1 / (df['Missed_Payments'] + 1)
df['Loan_Income_Ratio']  = df['Loan_Amount'] / df['Annual_Income']

feature_cols = [
    'Age', 'Annual_Income', 'Loan_Amount', 'Loan_Tenure_Months',
    'Existing_Debts', 'Missed_Payments', 'Employment_Years',
    'Debt_To_Income_Ratio', 'Credit_History_Enc', 'Loan_Purpose_Enc',
    'Income_Per_Debt', 'Payment_Reliability', 'Loan_Income_Ratio'
]

X = df[feature_cols]
y = df['Creditworthy']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE — Train size: {X_train_res.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

print(f"   Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")

# ── 5. MODEL TRAINING ─────────────────────────────────────────
print("\n\n🤖 Model Training")
print("-" * 40)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_sc, y_train_res)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    cv_scores = cross_val_score(model, X_train_sc, y_train_res, cv=skf, scoring='roc_auc')

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
        'cv_auc':    cv_scores.mean()
    }
    print(f"\n   {name}")
    print(f"     Accuracy : {results[name]['accuracy']:.4f}")
    print(f"     Precision: {results[name]['precision']:.4f}")
    print(f"     Recall   : {results[name]['recall']:.4f}")
    print(f"     F1-Score : {results[name]['f1']:.4f}")
    print(f"     ROC-AUC  : {results[name]['roc_auc']:.4f}")
    print(f"     CV AUC   : {results[name]['cv_auc']:.4f}")

# ── 6. BEST MODEL ────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['roc_auc'])
best = results[best_name]
print(f"\n\n🏆 Best Model: {best_name} (ROC-AUC = {best['roc_auc']:.4f})")

# ── 7. EVALUATION PLOTS ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Model Evaluation — Best: {best_name}', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not CW','Creditworthy'],
            yticklabels=['Not CW','Creditworthy'])
axes[0].set_title(f'Confusion Matrix\n{best_name}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curves — all models
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} ({res['roc_auc']:.3f})")
axes[1].plot([0,1],[0,1],'k--', label='Random')
axes[1].set_title('ROC Curves — All Models')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Metrics comparison bar chart
metric_names = ['Accuracy','Precision','Recall','F1','ROC-AUC']
x = np.arange(len(metric_names))
width = 0.2
colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']
for i, (name, res) in enumerate(results.items()):
    vals = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['roc_auc']]
    axes[2].bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
axes[2].set_xticks(x + width*1.5)
axes[2].set_xticklabels(metric_names, rotation=15)
axes[2].set_ylim(0.5, 1.05)
axes[2].set_title('Metrics Comparison — All Models')
axes[2].legend(fontsize=7)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Evaluation plots saved → outputs/model_evaluation.png")

# ── 8. FEATURE IMPORTANCE (Random Forest) ────────────────────
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(importances.index, importances.values,
                color=plt.cm.RdYlGn(importances.values / importances.values.max()))
plt.title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Feature importance saved → outputs/feature_importance.png")

# ── 9. CLASSIFICATION REPORT ─────────────────────────────────
print(f"\n\n📋 Classification Report — {best_name}")
print("-" * 40)
print(classification_report(y_test, best['y_pred'],
                             target_names=['Not Creditworthy','Creditworthy']))

# ── 10. PREDICT ON NEW APPLICANT ─────────────────────────────
print("\n\n🔍 Sample Prediction — New Loan Applicant")
print("-" * 40)

new_applicant = pd.DataFrame([{
    'Age': 35,
    'Annual_Income': 65000,
    'Loan_Amount': 15000,
    'Loan_Tenure_Months': 24,
    'Existing_Debts': 1,
    'Missed_Payments': 1,
    'Employment_Years': 8,
    'Debt_To_Income_Ratio': round(15000/65000, 4),
    'Credit_History_Enc': 2,   # Good
    'Loan_Purpose_Enc': 0,     # Car
    'Income_Per_Debt': 65000 / (1 + 1),
    'Payment_Reliability': 1 / (1 + 1),
    'Loan_Income_Ratio': 15000 / 65000
}])

new_scaled = scaler.transform(new_applicant)
pred  = best['model'].predict(new_scaled)[0]
prob  = best['model'].predict_proba(new_scaled)[0][1]

print(f"   Applicant Details: Age=35, Income=₹65,000, Loan=₹15,000")
print(f"   Missed Payments=1, Employment=8 yrs, Credit History=Good")
print(f"\n   ➤ Prediction  : {'✅ CREDITWORTHY' if pred == 1 else '❌ NOT CREDITWORTHY'}")
print(f"   ➤ Confidence  : {prob*100:.1f}%")

print("\n\n" + "=" * 60)
print("   ✅ Task 1 Complete! All outputs saved in /outputs/")
print("=" * 60)
