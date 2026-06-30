# 🏥 Disease Prediction from Medical Data — CodeAlpha Internship Task 4

Predicts the likelihood of diseases using patient medical data and classical ML classification algorithms across **3 real-world datasets**.

---

## 📌 Task Overview
| Item | Detail |
|------|--------|
| Internship | CodeAlpha — Machine Learning |
| Task | Task 4: Disease Prediction from Medical Data |
| Author | Shashank Singh |
| GitHub | [@Shashank17singh](https://github.com/Shashank17singh) |

---

## 📊 Datasets Used
| Dataset | Source | Samples | Task |
|---------|--------|---------|------|
| Heart Disease | UCI Cleveland | 303 | Binary classification |
| Diabetes | Pima Indians (UCI) | 768 | Binary classification |
| Breast Cancer | sklearn built-in | 569 | Binary classification |

---

## 🤖 Models & Results

### Heart Disease
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | 82.0% | 0.836 | 0.892 |
| Decision Tree | 73.8% | 0.714 | 0.749 |
| Random Forest | 85.2% | 0.862 | 0.891 |
| SVM | 77.0% | 0.788 | 0.854 |
| **XGBoost ✅** | **85.2%** | **0.842** | **0.911** |

### Diabetes
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | 70.8% | 0.545 | 0.815 |
| Decision Tree | 73.4% | 0.559 | 0.792 |
| **Random Forest ✅** | **76.0%** | **0.634** | **0.820** |
| SVM | 73.4% | 0.586 | 0.794 |
| XGBoost | 73.4% | 0.610 | 0.810 |

### Breast Cancer
| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| **Logistic Regression ✅** | **98.2%** | **0.986** | **0.995** |
| Decision Tree | 91.2% | 0.929 | 0.915 |
| Random Forest | 95.6% | 0.966 | 0.994 |
| SVM | 98.2% | 0.986 | 0.995 |
| XGBoost | 95.6% | 0.966 | 0.990 |

---

## 🔧 Tech Stack
- **Python 3**
- **Scikit-learn** — ML models, preprocessing, metrics
- **XGBoost** — Gradient boosting classifier
- **Pandas / NumPy** — Data handling
- **Matplotlib / Seaborn** — Visualizations

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Shashank17singh/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction

# Install dependencies
pip install -r requirements.txt

# Run the model
python disease_prediction.py
```

---

## 📁 Project Structure
```
CodeAlpha_DiseasePrediction/
│
├── disease_prediction.py      # Main script
├── requirements.txt           # Dependencies
├── README.md                  # Project docs
└── outputs/
    ├── roc_curves.png         # ROC curves — all models × all datasets
    ├── confusion_matrices.png # Confusion matrices — best models
    ├── metrics_heatmap.png    # AUC heatmap — all models × datasets
    └── feature_importance.png # Top features per dataset
```

---

## 📈 Output Plots
- **ROC Curves** — All 5 models compared across 3 datasets
- **Confusion Matrices** — Best model per dataset
- **Metrics Heatmap** — ROC-AUC scores across all models & datasets
- **Feature Importance** — Top features driving each disease prediction

---

## 📜 License
MIT License
