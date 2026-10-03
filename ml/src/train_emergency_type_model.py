"""
Multi-Label Emergency Type Classification Baseline Model Trainer
Hyperlocal Emergency Response Platform - Machine Learning Workspace

Pipeline: text -> TF-IDF vectorization -> OneVsRestClassifier(LogisticRegression(class_weight='balanced'))
Target: 35 emergency disaster/incident type categories parsed from `class_labels`.
Binary target `emergency` is excluded and kept separate.
"""

import ast
import os
import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# Suppress optimizer warnings & configure UTF-8 output
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_emergency_type_model():
    print("=" * 80)
    print(" TRAINING MULTI-LABEL EMERGENCY TYPE CLASSIFICATION BASELINE MODEL")
    print("=" * 80)

    # 1. Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    train_path = project_root / "ml" / "data" / "processed" / "train.csv"
    test_path = project_root / "ml" / "data" / "processed" / "test.csv"
    models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = models_dir / "emergency_type_classifier.joblib"
    labels_save_path = models_dir / "emergency_type_labels.joblib"
    report_save_path = models_dir / "emergency_type_model_report.txt"

    # 2. Load processed datasets
    print("\n[Step 1] Loading processed datasets...")
    print(f"  Train path: {train_path}")
    print(f"  Test path : {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 3. Input features
    X_train = train_df["text"].fillna("")
    X_test = test_df["text"].fillna("")

    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples    : {len(X_test):,}")

    # 4. Identify 35 emergency type labels (excluding 'emergency')
    candidate_cols = [c for c in train_df.columns if c not in ["text", "emergency", "class_labels"]]
    label_names = sorted(candidate_cols)
    num_labels = len(label_names)
    print(f"  Identified {num_labels} distinct emergency type categories (excluding binary 'emergency').")

    # 5. Parse `class_labels` into multi-label target matrices
    print("\n[Step 2] Parsing `class_labels` into multi-label target matrices...")

    def parse_labels(series):
        parsed = []
        for item in series:
            if isinstance(item, str):
                try:
                    labels = ast.literal_eval(item)
                except Exception:
                    labels = []
            elif isinstance(item, (list, set)):
                labels = list(item)
            else:
                labels = []
            # Exclude binary detection target 'emergency'
            cleaned = [lbl for lbl in labels if lbl != "emergency" and lbl in label_names]
            parsed.append(cleaned)
        return parsed

    parsed_train = parse_labels(train_df["class_labels"])
    parsed_test = parse_labels(test_df["class_labels"])

    mlb = MultiLabelBinarizer(classes=label_names)
    y_train = mlb.fit_transform(parsed_train)
    y_test = mlb.transform(parsed_test)

    print(f"  y_train target shape: {y_train.shape}")
    print(f"  y_test target shape : {y_test.shape}")

    # 6. Define Architecture: TF-IDF -> OneVsRestClassifier(LogisticRegression)
    print("\n[Step 3] Building TF-IDF + OneVsRest Logistic Regression pipeline...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    base_lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        class_weight="balanced",
    )
    ovr_clf = OneVsRestClassifier(base_lr, n_jobs=-1)

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", ovr_clf),
    ])

    # 7. Train pipeline
    print("\n[Step 4] Fitting multi-label pipeline on training set (35 binary classifiers)...")
    pipeline.fit(X_train, y_train)
    print("  Training finished successfully.")

    # Attach label metadata to clf estimator and pipeline object
    pipeline.named_steps["clf"].classes_ = np.array(label_names)
    pipeline.custom_label_names_ = label_names

    # 8. Evaluation on unseen test set
    print("\n[Step 5] Evaluating model on unseen test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # Overall metrics
    micro_prec = precision_score(y_test, y_pred, average="micro", zero_division=0)
    micro_rec = recall_score(y_test, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)

    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    hamm_loss = hamming_loss(y_test, y_pred)

    print(f"  Micro Precision: {micro_prec:.4f}")
    print(f"  Micro Recall   : {micro_rec:.4f}")
    print(f"  Micro F1       : {micro_f1:.4f}")
    print(f"  Macro Precision: {macro_prec:.4f}")
    print(f"  Macro Recall   : {macro_rec:.4f}")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Hamming Loss   : {hamm_loss:.4f} (bit error rate: {hamm_loss * 100:.2f}%)")

    # Per-label evaluation
    per_label_metrics = []
    mcm = multilabel_confusion_matrix(y_test, y_pred)

    for i, label in enumerate(label_names):
        p = precision_score(y_test[:, i], y_pred[:, i], average="binary", zero_division=0)
        r = recall_score(y_test[:, i], y_pred[:, i], average="binary", zero_division=0)
        f = f1_score(y_test[:, i], y_pred[:, i], average="binary", zero_division=0)
        test_supp = int(y_test[:, i].sum())
        train_supp = int(y_train[:, i].sum())
        tn, fp, fn, tp = mcm[i].ravel()
        per_label_metrics.append({
            "label": label,
            "train_support": train_supp,
            "test_support": test_supp,
            "precision": p,
            "recall": r,
            "f1": f,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    # Sort by test support descending for evaluation summaries
    sorted_by_support = sorted(per_label_metrics, key=lambda x: x["test_support"], reverse=True)

    # 9. Top frequent labels confusion summary
    top_frequent = sorted_by_support[:10]
    print("\n[Step 6] Top 10 Most Frequent Labels Confusion Summary:")
    print(f"  {'Label':<25} | {'Test Supp':<10} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}")
    print("  " + "-" * 95)
    for m in top_frequent:
        print(f"  {m['label']:<25} | {m['test_support']:<10} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1']:<7.4f} | {m['tp']:<6} | {m['fp']:<6} | {m['fn']:<6} | {m['tn']:<6}")

    # 10. Rare labels analysis
    rare_labels = [m for m in sorted_by_support if m["test_support"] < 100]
    unrepresented_labels = [m for m in sorted_by_support if m["train_support"] == 0]

    # 11. Save model pipeline and label encoder
    print(f"\n[Step 7] Saving model pipeline to: {model_save_path}")
    joblib.dump(pipeline, model_save_path)

    print(f"  Saving label encoder mapping to: {labels_save_path}")
    label_metadata = {
        "classes": label_names,
        "mlb": mlb,
        "num_classes": num_labels,
    }
    joblib.dump(label_metadata, labels_save_path)

    # 12. Generate and save comprehensive text report
    print(f"  Generating evaluation report at: {report_save_path}")
    report_lines = [
        "=" * 85,
        " HYPERLOCAL EMERGENCY RESPONSE PLATFORM - MULTI-LABEL EMERGENCY TYPE REPORT",
        "=" * 85,
        f"Training samples: {len(X_train):,}",
        f"Testing samples : {len(X_test):,}",
        f"Number of emergency type labels: {num_labels}",
        "Architecture    : Text -> TF-IDF (15,000 max_features, n-gram 1-2) -> OneVsRest LogisticRegression (balanced)",
        "Binary target   : 'emergency' is kept strictly separate for binary detection.",
        "",
        "-" * 85,
        " OVERALL EVALUATION METRICS",
        "-" * 85,
        f"Micro Precision : {micro_prec:.4f}",
        f"Micro Recall    : {micro_rec:.4f}",
        f"Micro F1-Score  : {micro_f1:.4f}",
        f"Macro Precision : {macro_prec:.4f}",
        f"Macro Recall    : {macro_rec:.4f}",
        f"Macro F1-Score  : {macro_f1:.4f}",
        f"Hamming Loss    : {hamm_loss:.4f} (Accuracy across label bits: {(1 - hamm_loss) * 100:.2f}%)",
        "",
        "-" * 85,
        " CONFUSION / EVALUATION SUMMARY FOR MOST FREQUENT LABELS (TOP 12)",
        "-" * 85,
        f"{'Label':<25} | {'Test Supp':<10} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}",
        "-" * 85,
    ]

    for m in sorted_by_support[:12]:
        report_lines.append(
            f"{m['label']:<25} | {m['test_support']:<10} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1']:<7.4f} | {m['tp']:<6} | {m['fp']:<6} | {m['fn']:<6} | {m['tn']:<6}"
        )

    report_lines.extend([
        "",
        "-" * 85,
        " PER-LABEL PERFORMANCE FOR ALL 35 EMERGENCY TYPES",
        "-" * 85,
        f"{'Index':<6} | {'Label':<25} | {'Train Supp':<11} | {'Test Supp':<10} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}",
        "-" * 85,
    ])

    for i, m in enumerate(sorted_by_support, 1):
        report_lines.append(
            f"{i:<6} | {m['label']:<25} | {m['train_support']:<11} | {m['test_support']:<10} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1']:<7.4f}"
        )

    report_lines.extend([
        "",
        "-" * 85,
        " CLASS IMBALANCE & RARE LABELS ANALYSIS",
        "-" * 85,
        f"1. Extreme Imbalance: Positive frequencies range from {sorted_by_support[0]['train_support']} ('{sorted_by_support[0]['label']}') down to 0 ('child_alone').",
        f"2. Completely Unrepresented Labels (0 training instances): {[m['label'] for m in unrepresented_labels]}",
        f"3. Very Rare Labels (<100 test samples): {[m['label'] for m in rare_labels]}",
        "4. Observations: Class balancing (class_weight='balanced') forces higher recall on rare classes at the cost of precision.",
        "",
        "-" * 85,
        " SCIKIT-LEARN MULTILABEL CLASSIFICATION REPORT",
        "-" * 85,
        classification_report(y_test, y_pred, target_names=label_names, zero_division=0),
        "",
        "-" * 85,
        " DISCLAIMER",
        "-" * 85,
        "This model is an experimental ML baseline for the Hyperlocal Emergency Response Platform.",
        "It is NOT verified or approved for real-world emergency dispatch or clinical safety.",
        "=" * 85,
    ])

    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n[Step 8] Complete! Report saved to: {report_save_path}")
    print("=" * 80)

    return {
        "num_labels": num_labels,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": hamm_loss,
        "model_path": str(model_save_path),
        "report_path": str(report_save_path),
    }


if __name__ == "__main__":
    train_emergency_type_model()
