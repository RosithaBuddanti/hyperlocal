"""
Binary Emergency Detection Baseline Model Trainer
Hyperlocal Emergency Response Platform - Machine Learning Workspace

Pipeline: text -> TF-IDF vectorization -> Logistic Regression classifier
Target: emergency (0 = non-emergency, 1 = emergency)
"""

import os
import sys
import warnings
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# Suppress optimizer warnings & configure UTF-8 output
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_emergency_model():
    print("=" * 80)
    print(" TRAINING BINARY EMERGENCY DETECTION BASELINE MODEL")
    print("=" * 80)

    # 1. Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    train_path = project_root / "ml" / "data" / "processed" / "train.csv"
    test_path = project_root / "ml" / "data" / "processed" / "test.csv"
    models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = models_dir / "emergency_classifier.joblib"
    report_save_path = models_dir / "emergency_model_report.txt"

    # 2. Load processed datasets
    print(f"\n[Step 1] Loading processed datasets...")
    print(f"  Train: {train_path}")
    print(f"  Test : {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["text"].fillna("")
    y_train = train_df["emergency"].astype(int)

    X_test = test_df["text"].fillna("")
    y_test = test_df["emergency"].astype(int)

    print(f"  Loaded {len(X_train):,} training examples.")
    print(f"  Loaded {len(X_test):,} test examples.")
    print(f"  Train class distribution: Non-Emergency (0)={sum(y_train==0):,}, Emergency (1)={sum(y_train==1):,}")
    print(f"  Test class distribution : Non-Emergency (0)={sum(y_test==0):,}, Emergency (1)={sum(y_test==1):,}")

    # 3. Define Pipeline (TfidfVectorizer -> LogisticRegression)
    print("\n[Step 2] Building TF-IDF + Logistic Regression pipeline...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])

    # 4. Train pipeline
    print("\n[Step 3] Fitting model on training data...")
    pipeline.fit(X_train, y_train)
    print("  Model training complete.")

    # 5. Evaluate on test set
    print("\n[Step 4] Evaluating model on unseen test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    prec_binary = precision_score(y_test, y_pred, average="binary")
    rec_binary = recall_score(y_test, y_pred, average="binary")
    f1_binary = f1_score(y_test, y_pred, average="binary")

    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    cls_report = classification_report(
        y_test,
        y_pred,
        target_names=["non-emergency", "emergency"],
        digits=4,
    )

    # 6. Save full pipeline
    print(f"\n[Step 5] Saving complete trained pipeline...")
    joblib.dump(pipeline, model_save_path)
    print(f"  -> Saved pipeline to: {model_save_path} ({os.path.getsize(model_save_path):,} bytes)")

    # 7. Write comprehensive report file
    report_text = f"""================================================================================
 HYPERLOCAL EMERGENCY RESPONSE PLATFORM - EMERGENCY DETECTION MODEL REPORT
================================================================================

1. MODEL OVERVIEW:
   - Model Type: TF-IDF Vectorizer + Logistic Regression Classifier
   - Pipeline  : text -> TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
                        -> LogisticRegression(max_iter=1000, C=1.0, random_state=42)
   - Task      : Binary Emergency Detection
   - Classes   : 0 = non-emergency, 1 = emergency
   - Note      : This baseline model detects emergency vs non-emergency messages only.
                 It does NOT predict emergency severity or multi-label categories.

2. DATASET SUMMARY:
   - Training Set : {len(X_train):,} samples (Non-Emergency: {sum(y_train==0):,}, Emergency: {sum(y_train==1):,})
   - Test Set     : {len(X_test):,} samples (Non-Emergency: {sum(y_test==0):,}, Emergency: {sum(y_test==1):,})

3. EVALUATION METRICS (TEST SET):
   - Overall Accuracy  : {acc * 100:.2f}% ({acc:.4f})
   - Emergency (Class 1) Precision: {prec_binary:.4f} ({prec_binary * 100:.2f}%)
   - Emergency (Class 1) Recall   : {rec_binary:.4f} ({rec_binary * 100:.2f}%)
   - Emergency (Class 1) F1-Score : {f1_binary:.4f} ({f1_binary * 100:.2f}%)
   - Macro Avg Precision          : {prec_macro:.4f} ({prec_macro * 100:.2f}%)
   - Macro Avg Recall             : {rec_macro:.4f} ({rec_macro * 100:.2f}%)
   - Macro Avg F1-Score           : {f1_macro:.4f} ({f1_macro * 100:.2f}%)

4. CONFUSION MATRIX:
               Predicted Non-Emergency (0) | Predicted Emergency (1)
   Actual Non-Emergency (0):    {cm[0, 0]:<15d} | {cm[0, 1]:<15d}
   Actual Emergency (1)    :    {cm[1, 0]:<15d} | {cm[1, 1]:<15d}

   - True Negatives  (Non-Emergency correctly identified): {cm[0, 0]:,d}
   - False Positives (Non-Emergency flagged as Emergency): {cm[0, 1]:,d}
   - False Negatives (Emergency missed by model)         : {cm[1, 0]:,d}
   - True Positives  (Emergency correctly identified)    : {cm[1, 1]:,d}

5. DETAILED CLASSIFICATION REPORT:
{cls_report}

6. SAVED ARTIFACTS:
   - Model Pipeline: {model_save_path}
   - Report File   : {report_save_path}
================================================================================
"""

    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  -> Saved report to: {report_save_path}")

    # 8. Print report highlights to stdout
    print("\n" + "=" * 80)
    print(" EVALUATION RESULTS (TEST SET)")
    print("=" * 80)
    print(f"Accuracy         : {acc * 100:.2f}%")
    print(f"Precision (Class 1 Emergency): {prec_binary:.4f}")
    print(f"Recall    (Class 1 Emergency): {rec_binary:.4f}")
    print(f"F1-Score  (Class 1 Emergency): {f1_binary:.4f}")
    print(f"Macro F1-Score               : {f1_macro:.4f}")
    print("\nConfusion Matrix:")
    print(f"  [[TN={cm[0, 0]}, FP={cm[0, 1]}],")
    print(f"   [FN={cm[1, 0]}, TP={cm[1, 1]}]]")
    print("\nClassification Report:")
    print(cls_report)
    print("=" * 80)
    print(" [OK] EMERGENCY MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    train_emergency_model()
