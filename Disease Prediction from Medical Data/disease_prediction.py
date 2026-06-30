# =============================================================
# CodeAlpha Internship — Task 4: Disease Prediction from Medical Data
# Author: Shashank Singh
# GitHub: https://github.com/Shashank17singh
# Datasets: Heart Disease, Diabetes, Breast Cancer (UCI / sklearn)
# =============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import os
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("   CodeAlpha ML — Task 4: Disease Prediction from Medical Data")
print("=" * 65)

# ── 2. DATASETS ──────────────────────────────────────────────

# ---- 2a. Heart Disease (Cleveland UCI) ----------------------
def load_heart_disease():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    try:
        df = pd.read_csv(url, header=None, na_values='?')
        df.columns = ['age','sex','cp','trestbps','chol','fbs',
                      'restecg','thalach','exang','oldpeak',
                      'slope','ca','thal','target']
        df.dropna(inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
        return df
    except:
        # Fallback synthetic if network unavailable
        np.random.seed(0)
        n = 303
        df = pd.DataFrame({
            'age':      np.random.randint(29,78,n),
            'sex':      np.random.randint(0,2,n),
            'cp':       np.random.randint(0,4,n),
            'trestbps': np.random.randint(94,200,n),
            'chol':     np.random.randint(126,565,n),
            'fbs':      np.random.randint(0,2,n),
            'restecg':  np.random.randint(0,3,n),
            'thalach':  np.random.randint(71,202,n),
            'exang':    np.random.randint(0,2,n),
            'oldpeak':  np.round(np.random.uniform(0,6.2,n),1),
            'slope':    np.random.randint(0,3,n),
            'ca':       np.random.randint(0,4,n),
            'thal':     np.random.randint(0,3,n),
        })
        score = (df['age']>55)*1 + (df['cp']==0)*2 + (df['thalach']<140)*2 + \
                (df['exang']==1)*2 + (df['oldpeak']>2)*1
        df['target'] = (score + np.random.normal(0,1,n) > score.mean()).astype(int)
        return df

# ---- 2b. Diabetes (Pima Indians style) ----------------------
def load_diabetes():
    try:
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
               "master/pima-indians-diabetes.data.csv")
        cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
        df = pd.read_csv(url, header=None, names=cols)
        # Replace 0s in medical columns with median
        for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
            df[c] = df[c].replace(0, df[c].median())
        return df
    except:
        np.random.seed(1)
        n = 768
        df = pd.DataFrame({
            'Pregnancies':              np.random.randint(0,18,n),
            'Glucose':                  np.random.randint(56,200,n),
            'BloodPressure':            np.random.randint(24,122,n),
            'SkinThickness':            np.random.randint(7,99,n),
            'Insulin':                  np.random.randint(14,846,n),
            'BMI':                      np.round(np.random.uniform(18,67,n),1),
            'DiabetesPedigreeFunction': np.round(np.random.uniform(0.08,2.42,n),3),
            'Age':                      np.random.randint(21,82,n),
        })
        score = (df['Glucose']>140)*3 + (df['BMI']>30)*2 + \
                (df['Age']>45)*1 + (df['Pregnancies']>3)*1
        df['Outcome'] = (score + np.random.normal(0,1,n) > score.mean()).astype(int)
        return df

# ---- 2c. Breast Cancer (sklearn built-in) -------------------
def load_breast_cancer_df():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target   # 1=benign, 0=malignant
    return df

# ── 3. MODELS ────────────────────────────────────────────────
def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM':                 SVC(probability=True, kernel='rbf', random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric='logloss', verbosity=0)
    }

# ── 4. TRAINING & EVALUATION PIPELINE ────────────────────────
def train_evaluate(X, y, dataset_name):
    print(f"\n{'─'*65}")
    print(f"  📂 Dataset: {dataset_name}  |  Samples: {len(y)}  |  "
          f"Positive: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"{'─'*65}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in get_models().items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        cv     = cross_val_score(model, X_train_sc, y_train,
                                 cv=skf, scoring='roc_auc').mean()
        results[name] = {
            'model':     model,
            'scaler':    scaler,
            'y_pred':    y_pred,
            'y_prob':    y_prob,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_prob),
            'cv_auc':    cv,
            'y_test':    y_test,
            'X_test_sc': X_test_sc,
        }
        print(f"  {name:<22} Acc={results[name]['accuracy']:.3f}  "
              f"F1={results[name]['f1']:.3f}  "
              f"AUC={results[name]['roc_auc']:.3f}  "
              f"CV-AUC={cv:.3f}")

    best = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\n  🏆 Best: {best}  (ROC-AUC = {results[best]['roc_auc']:.4f})")
    return results, best

# ── 5. RUN ALL THREE DATASETS ─────────────────────────────────
print("\n🔄 Loading datasets...")
heart_df    = load_heart_disease()
diabetes_df = load_diabetes()
cancer_df   = load_breast_cancer_df()
print("  ✅ All datasets loaded")

datasets = {
    'Heart Disease': (
        heart_df.drop('target', axis=1),
        heart_df['target']
    ),
    'Diabetes': (
        diabetes_df.drop('Outcome', axis=1),
        diabetes_df['Outcome']
    ),
    'Breast Cancer': (
        cancer_df.drop('target', axis=1),
        cancer_df['target']
    ),
}

all_results = {}
all_best    = {}
for dname, (X, y) in datasets.items():
    res, best = train_evaluate(X, y, dname)
    all_results[dname] = res
    all_best[dname]    = best

# ── 6. PLOT 1 — ROC CURVES (3 datasets × 5 models) ───────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ROC Curves — All Datasets & Models', fontsize=15, fontweight='bold')

colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
model_names = list(get_models().keys())

for ax, (dname, res) in zip(axes, all_results.items()):
    for (mname, mres), color in zip(res.items(), colors):
        fpr, tpr, _ = roc_curve(mres['y_test'], mres['y_prob'])
        ax.plot(fpr, tpr, color=color,
                label=f"{mname} ({mres['roc_auc']:.3f})", linewidth=1.8)
    ax.plot([0,1],[0,1],'k--', linewidth=1)
    ax.set_title(dname, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n\n✅ ROC curves saved → outputs/roc_curves.png")

# ── 7. PLOT 2 — CONFUSION MATRICES (best model per dataset) ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices — Best Model per Dataset',
             fontsize=14, fontweight='bold')

for ax, (dname, res) in zip(axes, all_results.items()):
    best = all_best[dname]
    bres = res[best]
    cm   = confusion_matrix(bres['y_test'], bres['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={'size': 14})
    ax.set_title(f"{dname}\n{best}", fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrices saved → outputs/confusion_matrices.png")

# ── 8. PLOT 3 — METRICS HEATMAP (all models × all datasets) ──
metric = 'roc_auc'
heatmap_data = pd.DataFrame(
    {dname: {mname: res[mname][metric] for mname in res}
     for dname, res in all_results.items()}
)

plt.figure(figsize=(9, 5))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGn',
            linewidths=0.5, annot_kws={'size': 12},
            vmin=0.7, vmax=1.0)
plt.title('ROC-AUC Scores — All Models × All Datasets',
          fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('outputs/metrics_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Metrics heatmap saved → outputs/metrics_heatmap.png")

# ── 9. PLOT 4 — FEATURE IMPORTANCE (Random Forest per dataset) ─
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Feature Importance — Random Forest per Dataset',
             fontsize=14, fontweight='bold')

dataset_cols = {
    'Heart Disease': heart_df.drop('target', axis=1).columns.tolist(),
    'Diabetes':      diabetes_df.drop('Outcome', axis=1).columns.tolist(),
    'Breast Cancer': cancer_df.drop('target', axis=1).columns.tolist(),
}

for ax, (dname, res) in zip(axes, all_results.items()):
    rf_model = res['Random Forest']['model']
    cols     = dataset_cols[dname]
    imp      = pd.Series(rf_model.feature_importances_, index=cols).sort_values()
    top      = imp.tail(10)
    bars = ax.barh(top.index, top.values,
                   color=plt.cm.RdYlGn(top.values / top.values.max()))
    ax.set_title(dname, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Feature importance saved → outputs/feature_importance.png")

# ── 10. SUMMARY TABLE ────────────────────────────────────────
print("\n\n" + "=" * 65)
print("   📊 FINAL SUMMARY — Best Model per Dataset")
print("=" * 65)
print(f"  {'Dataset':<18} {'Best Model':<24} {'Acc':>6} {'F1':>6} {'AUC':>6}")
print("  " + "-" * 62)
for dname, res in all_results.items():
    best  = all_best[dname]
    bres  = res[best]
    print(f"  {dname:<18} {best:<24} "
          f"{bres['accuracy']:>6.3f} {bres['f1']:>6.3f} {bres['roc_auc']:>6.3f}")

# ── 11. SAMPLE PREDICTION ────────────────────────────────────
print("\n\n🔍 Sample Prediction — Heart Disease")
print("-" * 40)
hd_res   = all_results['Heart Disease']
hd_best  = all_best['Heart Disease']
hd_model = hd_res[hd_best]['model']
hd_scaler= hd_res[hd_best]['scaler']

sample = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
sample_sc = hd_scaler.transform(sample)
pred  = hd_model.predict(sample_sc)[0]
prob  = hd_model.predict_proba(sample_sc)[0][1]
print(f"  Patient: Age=63, Male, Chest Pain Type=3, Chol=233")
print(f"  ➤ Prediction : {'❤️  HEART DISEASE DETECTED' if pred==1 else '✅ NO HEART DISEASE'}")
print(f"  ➤ Confidence : {prob*100:.1f}%")

print("\n\n" + "=" * 65)
print("   ✅ Task 4 Complete! All outputs saved in /outputs/")
print("=" * 65)
