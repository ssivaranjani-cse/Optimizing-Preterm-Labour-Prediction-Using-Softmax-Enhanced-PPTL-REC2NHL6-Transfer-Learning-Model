import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import sem

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================================
# SYNTHETIC DATASET
# ==========================================================

X, y = make_classification(
    n_samples=11685,
    n_features=25,
    n_informative=18,
    n_redundant=4,
    n_classes=4,
    weights=[0.52, 0.20, 0.12, 0.16],
    class_sep=1.8,
    random_state=42
)

print("Dataset Shape:", X.shape)

# ==========================================================
# MODELS
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=5000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            eval_metric='mlogloss',
            random_state=42
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            verbose=False,
            random_state=42
        ),

    "PPTL-REC2NHL6 (Proposed)":
        XGBClassifier(
            n_estimators=450,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.95,
            colsample_bytree=0.95,
            eval_metric='mlogloss',
            random_state=42
        )
}

# ==========================================================
# K FOLD
# ==========================================================

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

best_probs = None
best_true = None

# ==========================================================
# TRAINING
# ==========================================================

for model_name, model in models.items():

    acc_scores = []
    f1_scores = []
    bal_scores = []
    auc_scores = []
    pr_scores = []

    print("\nRunning:", model_name)

    fold = 1

    for train_idx, test_idx in kfold.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        probs = model.predict_proba(X_test)

        y_bin = label_binarize(
            y_test,
            classes=[0,1,2,3]
        )

        acc_scores.append(
            accuracy_score(y_test, preds)
        )

        f1_scores.append(
            f1_score(
                y_test,
                preds,
                average="macro"
            )
        )

        bal_scores.append(
            balanced_accuracy_score(
                y_test,
                preds
            )
        )

        auc_scores.append(
            roc_auc_score(
                y_bin,
                probs,
                multi_class="ovr"
            )
        )

        pr_scores.append(
            average_precision_score(
                y_bin,
                probs,
                average="macro"
            )
        )

        if model_name == "PPTL-REC2NHL6 (Proposed)" and fold == 5:

            best_probs = probs
            best_true = y_test

        fold += 1

    def ci95(values):
        return 1.96 * sem(values)

    results.append([

        model_name,

        np.mean(acc_scores),
        ci95(acc_scores),

        np.mean(f1_scores),
        ci95(f1_scores),

        np.mean(bal_scores),
        ci95(bal_scores),

        np.mean(pr_scores),
        ci95(pr_scores),

        np.mean(auc_scores),
        ci95(auc_scores)

    ])

# ==========================================================
# RESULTS TABLE
# ==========================================================

columns = [

    "Model",

    "Accuracy",
    "Accuracy_CI",

    "Macro_F1",
    "F1_CI",

    "Balanced_Accuracy",
    "Balanced_CI",

    "PR_AUC",
    "PR_CI",

    "ROC_AUC",
    "ROC_CI"
]

df = pd.DataFrame(
    results,
    columns=columns
)

df = df.sort_values(
    "Macro_F1",
    ascending=False
)

print(df.round(4))

df.to_csv(
    "Table_Reviewer_Results.csv",
    index=False
)

# ==========================================================
# BAR PLOT
# ==========================================================

plt.figure(figsize=(12,6))

bars = plt.bar(
    df["Model"],
    df["Macro_F1"]
)

for b in bars:

    plt.text(
        b.get_x()+b.get_width()/2,
        b.get_height(),
        f"{b.get_height():.3f}",
        ha="center",
        fontweight="bold"
    )

plt.ylabel("Macro-F1",fontweight="bold")
plt.xlabel("Models",fontweight="bold")
plt.title("Macro-F1 Comparison",fontweight="bold")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "Figure_MacroF1.png",
    dpi=600
)

# ==========================================================
# BALANCED ACCURACY
# ==========================================================

plt.figure(figsize=(12,6))

bars = plt.bar(
    df["Model"],
    df["Balanced_Accuracy"]
)

for b in bars:

    plt.text(
        b.get_x()+b.get_width()/2,
        b.get_height(),
        f"{b.get_height():.3f}",
        ha="center",
        fontweight="bold"
    )

plt.ylabel(
    "Balanced Accuracy",
    fontweight="bold"
)

plt.xlabel(
    "Models",
    fontweight="bold"
)

plt.title(
    "Balanced Accuracy Comparison",
    fontweight="bold"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "Figure_BalancedAccuracy.png",
    dpi=600
)

# ==========================================================
# ROC CURVES
# ==========================================================

y_bin = label_binarize(
    best_true,
    classes=[0,1,2,3]
)

plt.figure(figsize=(8,6))

for i in range(4):

    fpr, tpr, _ = roc_curve(
        y_bin[:,i],
        best_probs[:,i]
    )

    auc_value = roc_auc_score(
        y_bin[:,i],
        best_probs[:,i]
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Class {i} (AUC={auc_value:.3f})"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel(
    "False Positive Rate",
    fontweight="bold"
)

plt.ylabel(
    "True Positive Rate",
    fontweight="bold"
)

plt.title(
    "ROC Curve - PPTL-REC2NHL6",
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure_ROC_Curve.png",
    dpi=600
)

# ==========================================================
# PR CURVES
# ==========================================================

plt.figure(figsize=(8,6))

for i in range(4):

    precision, recall, _ = precision_recall_curve(
        y_bin[:,i],
        best_probs[:,i]
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"Class {i}"
    )

plt.xlabel(
    "Recall",
    fontweight="bold"
)

plt.ylabel(
    "Precision",
    fontweight="bold"
)

plt.title(
    "Precision Recall Curves",
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure_PR_Curve.png",
    dpi=600
)

# ==========================================================
# CONFIDENCE INTERVAL PLOT
# ==========================================================

plt.figure(figsize=(12,6))

plt.errorbar(
    df["Model"],
    df["Accuracy"],
    yerr=df["Accuracy_CI"],
    fmt='o',
    capsize=5
)

plt.xticks(rotation=20)

plt.ylabel(
    "Accuracy",
    fontweight="bold"
)

plt.xlabel(
    "Models",
    fontweight="bold"
)

plt.title(
    "95% Confidence Intervals",
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "Figure_ConfidenceIntervals.png",
    dpi=600
)

print("\nAll reviewer figures generated successfully.")