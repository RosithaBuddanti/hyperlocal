"""
Emergency Detection Model Benchmark & Threshold Calibration
Hyperlocal Emergency Response Platform - Machine Learning Workspace

Compares:
- Experiment A: Baseline Logistic Regression (class_weight=None)
- Experiment B: Balanced Logistic Regression (class_weight="balanced")
Evaluates decision thresholds: [0.30, 0.40, 0.50, 0.60, 0.70]
Saves report to ml/models/emergency_benchmark_report.txt
Saves balanced model to ml/models/emergency_classifier_balanced.joblib
"""

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
    accuracy_score,
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


def evaluate_at_threshold(y_true, y_probs, threshold=0.5):
    """Computes binary classification metrics given a decision threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec_emerg = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_emerg = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_emerg = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    prec_non_emerg = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_non_emerg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_non_emerg = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "prec_emerg": prec_emerg,
        "rec_emerg": rec_emerg,
        "f1_emerg": f1_emerg,
        "prec_non_emerg": prec_non_emerg,
        "rec_non_emerg": rec_non_emerg,
        "f1_non_emerg": f1_non_emerg,
        "macro_f1": macro_f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run_benchmark():
    print("=" * 85)
    print(" HYPERLOCAL EMERGENCY PLATFORM - CLASSIFIER BENCHMARK & THRESHOLD ANALYSIS")
    print("=" * 85)

    # 1. Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    train_path = project_root / "ml" / "data" / "processed" / "train.csv"
    test_path = project_root / "ml" / "data" / "processed" / "test.csv"
    models_dir = project_root / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    report_save_path = models_dir / "emergency_benchmark_report.txt"
    balanced_model_save_path = models_dir / "emergency_classifier_balanced.joblib"

    # 2. Load datasets
    print("\n[Step 1] Loading processed train and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["text"].fillna("")
    y_train = train_df["emergency"].astype(int)

    X_test = test_df["text"].fillna("")
    y_test = test_df["emergency"].astype(int)

    print(f"  Train: {len(X_train):,} samples (Non-Emergency: {sum(y_train==0):,}, Emergency: {sum(y_train==1):,})")
    print(f"  Test : {len(X_test):,} samples (Non-Emergency: {sum(y_test==0):,}, Emergency: {sum(y_test==1):,})")

    # 3. Train Experiment A (Baseline: class_weight=None)
    print("\n[Step 2] Training Experiment A: Baseline Model (class_weight=None)...")
    pipe_baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight=None))
    ])
    pipe_baseline.fit(X_train, y_train)
    probs_baseline = pipe_baseline.predict_proba(X_test)[:, 1]

    # 4. Train Experiment B (Balanced: class_weight='balanced')
    print("[Step 3] Training Experiment B: Balanced Model (class_weight='balanced')...")
    pipe_balanced = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True, min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight="balanced"))
    ])
    pipe_balanced.fit(X_train, y_train)
    probs_balanced = pipe_balanced.predict_proba(X_test)[:, 1]

    # Save balanced model
    joblib.dump(pipe_balanced, balanced_model_save_path)
    print(f"  -> Saved balanced candidate model to: {balanced_model_save_path}")

    # 5. Evaluate both models at default 0.50 threshold
    res_baseline_default = evaluate_at_threshold(y_test, probs_baseline, 0.50)
    res_balanced_default = evaluate_at_threshold(y_test, probs_balanced, 0.50)

    # 6. Threshold sweeps
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    sweep_baseline = [evaluate_at_threshold(y_test, probs_baseline, t) for t in thresholds]
    sweep_balanced = [evaluate_at_threshold(y_test, probs_balanced, t) for t in thresholds]

    # 7. Benchmark sample predictions
    benchmark_messages = [
        "There is a major fire near the highway and people need help.",
        "I lost my wallet yesterday.",
        "A car has crashed and someone appears injured.",
        "There is heavy flooding and people are trapped.",
        "Can someone tell me today's weather?",
    ]

    preds_baseline_samples = []
    preds_balanced_samples = []

    for msg in benchmark_messages:
        prob_b0 = pipe_baseline.predict_proba([msg])[0]
        prob_b1 = pipe_balanced.predict_proba([msg])[0]

        preds_baseline_samples.append({
            "text": msg,
            "pred": "emergency (1)" if prob_b0[1] >= 0.5 else "non-emergency (0)",
            "p_emerg": prob_b0[1],
            "p_non_emerg": prob_b0[0],
        })

        preds_balanced_samples.append({
            "text": msg,
            "pred_0.5": "emergency (1)" if prob_b1[1] >= 0.5 else "non-emergency (0)",
            "pred_0.6": "emergency (1)" if prob_b1[1] >= 0.6 else "non-emergency (0)",
            "p_emerg": prob_b1[1],
            "p_non_emerg": prob_b1[0],
        })

    # 8. Generate comprehensive benchmark report
    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append(" HYPERLOCAL EMERGENCY PLATFORM - EMERGENCY CLASSIFIER BENCHMARK REPORT")
    report_lines.append("=" * 90)
    report_lines.append("")
    report_lines.append("1. OBJECTIVE & EXPERIMENTAL SETUP:")
    report_lines.append("   - Task        : Binary Emergency Detection (0 = non-emergency, 1 = emergency)")
    report_lines.append("   - Input       : Cleaned text messages from train.csv (20,749) and test.csv (5,196)")
    report_lines.append("   - Pipeline    : text -> TfidfVectorizer(max_features=15000, ngram_range=(1,2)) -> LogisticRegression")
    report_lines.append("   - Models Tested:")
    report_lines.append("     * Experiment A: class_weight=None (Baseline, preserves original class prior)")
    report_lines.append("     * Experiment B: class_weight='balanced' (Inversely proportional to class frequencies)")
    report_lines.append("   - Scope Limitation:")
    report_lines.append("     * This model strictly detects binary emergency occurrence.")
    report_lines.append("     * It does NOT estimate emergency severity.")
    report_lines.append("     * It does NOT perform multi-label agency/disaster categorization.")
    report_lines.append("     * Threshold calibration does NOT guarantee operational or clinical infallibility;")
    report_lines.append("       it illustrates the precision/recall trade-off for emergency dispatch operations.")
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("2. HEAD-TO-HEAD COMPARISON AT DEFAULT THRESHOLD (0.50):")
    report_lines.append("=" * 90)
    report_lines.append(f"   {'Metric':<28} | {'Exp A: Baseline (None)':<24} | {'Exp B: Balanced':<24}")
    report_lines.append("   " + "-" * 82)
    report_lines.append(f"   {'Overall Accuracy':<28} | {res_baseline_default['accuracy']*100:6.2f}%                  | {res_balanced_default['accuracy']*100:6.2f}%")
    report_lines.append(f"   {'Emergency Precision (1)':<28} | {res_baseline_default['prec_emerg']*100:6.2f}%                  | {res_balanced_default['prec_emerg']*100:6.2f}%")
    report_lines.append(f"   {'Emergency Recall (1)':<28} | {res_baseline_default['rec_emerg']*100:6.2f}%                  | {res_balanced_default['rec_emerg']*100:6.2f}%")
    report_lines.append(f"   {'Emergency F1-Score (1)':<28} | {res_baseline_default['f1_emerg']*100:6.2f}%                  | {res_balanced_default['f1_emerg']*100:6.2f}%")
    report_lines.append(f"   {'Non-Emergency Precision (0)':<28} | {res_baseline_default['prec_non_emerg']*100:6.2f}%                  | {res_balanced_default['prec_non_emerg']*100:6.2f}%")
    report_lines.append(f"   {'Non-Emergency Recall (0)':<28} | {res_baseline_default['rec_non_emerg']*100:6.2f}%                  | {res_balanced_default['rec_non_emerg']*100:6.2f}%")
    report_lines.append(f"   {'Non-Emergency F1-Score (0)':<28} | {res_baseline_default['f1_non_emerg']*100:6.2f}%                  | {res_balanced_default['f1_non_emerg']*100:6.2f}%")
    report_lines.append(f"   {'Macro Average F1':<28} | {res_baseline_default['macro_f1']*100:6.2f}%                  | {res_balanced_default['macro_f1']*100:6.2f}%")
    report_lines.append("   " + "-" * 82)
    report_lines.append(f"   {'True Negatives (TN)':<28} | {res_baseline_default['tn']:<24,d} | {res_balanced_default['tn']:<24,d}")
    report_lines.append(f"   {'False Positives (FP)':<28} | {res_baseline_default['fp']:<24,d} | {res_balanced_default['fp']:<24,d}")
    report_lines.append(f"   {'False Negatives (FN)':<28} | {res_baseline_default['fn']:<24,d} | {res_balanced_default['fn']:<24,d}")
    report_lines.append(f"   {'True Positives (TP)':<28} | {res_baseline_default['tp']:<24,d} | {res_balanced_default['tp']:<24,d}")
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("3. PROBABILITY THRESHOLD SWEEP - EXPERIMENT B (BALANCED MODEL):")
    report_lines.append("=" * 90)
    report_lines.append("   Threshold | Accuracy | Emerg Prec | Emerg Rec | Emerg F1 | Non-Em Prec | Non-Em Rec | Macro F1 |  FP  |  FN ")
    report_lines.append("   " + "-" * 95)
    for r in sweep_balanced:
        report_lines.append(
            f"     {r['threshold']:0.2f}    |  {r['accuracy']*100:5.2f}%  |   {r['prec_emerg']*100:5.2f}%   |  {r['rec_emerg']*100:5.2f}%  |  {r['f1_emerg']*100:5.2f}%  |"
            f"   {r['prec_non_emerg']*100:5.2f}%   |   {r['rec_non_emerg']*100:5.2f}%   |  {r['macro_f1']*100:5.2f}%  | {r['fp']:4d} | {r['fn']:4d}"
        )
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("4. PROBABILITY THRESHOLD SWEEP - EXPERIMENT A (BASELINE UNWEIGHTED MODEL):")
    report_lines.append("=" * 90)
    report_lines.append("   Threshold | Accuracy | Emerg Prec | Emerg Rec | Emerg F1 | Non-Em Prec | Non-Em Rec | Macro F1 |  FP  |  FN ")
    report_lines.append("   " + "-" * 95)
    for r in sweep_baseline:
        report_lines.append(
            f"     {r['threshold']:0.2f}    |  {r['accuracy']*100:5.2f}%  |   {r['prec_emerg']*100:5.2f}%   |  {r['rec_emerg']*100:5.2f}%  |  {r['f1_emerg']*100:5.2f}%  |"
            f"   {r['prec_non_emerg']*100:5.2f}%   |   {r['rec_non_emerg']*100:5.2f}%   |  {r['macro_f1']*100:5.2f}%  | {r['fp']:4d} | {r['fn']:4d}"
        )
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("5. BENCHMARK TEST SAMPLES EVALUATION:")
    report_lines.append("=" * 90)
    for i, (b_item, bal_item) in enumerate(zip(preds_baseline_samples, preds_balanced_samples), 1):
        report_lines.append(f"\n   [Sample #{i}]")
        report_lines.append(f"   Message: \"{b_item['text']}\"")
        report_lines.append(f"   - Exp A (Baseline, thr=0.50): {b_item['pred']} (P(emerg)={b_item['p_emerg']*100:5.2f}%, P(non-emerg)={b_item['p_non_emerg']*100:5.2f}%)")
        report_lines.append(f"   - Exp B (Balanced, thr=0.50): {bal_item['pred_0.5']} (P(emerg)={bal_item['p_emerg']*100:5.2f}%, P(non-emerg)={bal_item['p_non_emerg']*100:5.2f}%)")
        report_lines.append(f"   - Exp B (Balanced, thr=0.60): {bal_item['pred_0.6']}")
    report_lines.append("")
    report_lines.append("=" * 90)
    report_lines.append("6. SAVED ARTIFACTS:")
    report_lines.append(f"   - Baseline Model : {models_dir / 'emergency_classifier.joblib'} (Untouched)")
    report_lines.append(f"   - Balanced Model : {balanced_model_save_path}")
    report_lines.append(f"   - Benchmark Report: {report_save_path}")
    report_lines.append("=" * 90)

    report_text = "\n".join(report_lines)
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[Step 4] Report saved to: {report_save_path}")

    # Print summary tables to stdout
    print("\n" + "=" * 90)
    print(" HEAD-TO-HEAD COMPARISON TABLE (Threshold = 0.50)")
    print("=" * 90)
    print(f"{'Metric':<28} | {'Exp A: Baseline (None)':<24} | {'Exp B: Balanced':<24}")
    print("-" * 82)
    print(f"{'Overall Accuracy':<28} | {res_baseline_default['accuracy']*100:6.2f}%                  | {res_balanced_default['accuracy']*100:6.2f}%")
    print(f"{'Emergency Precision (1)':<28} | {res_baseline_default['prec_emerg']*100:6.2f}%                  | {res_balanced_default['prec_emerg']*100:6.2f}%")
    print(f"{'Emergency Recall (1)':<28} | {res_baseline_default['rec_emerg']*100:6.2f}%                  | {res_balanced_default['rec_emerg']*100:6.2f}%")
    print(f"{'Emergency F1-Score (1)':<28} | {res_baseline_default['f1_emerg']*100:6.2f}%                  | {res_balanced_default['f1_emerg']*100:6.2f}%")
    print(f"{'Non-Emergency Precision (0)':<28} | {res_baseline_default['prec_non_emerg']*100:6.2f}%                  | {res_balanced_default['prec_non_emerg']*100:6.2f}%")
    print(f"{'Non-Emergency Recall (0)':<28} | {res_baseline_default['rec_non_emerg']*100:6.2f}%                  | {res_balanced_default['rec_non_emerg']*100:6.2f}%")
    print(f"{'Non-Emergency F1-Score (0)':<28} | {res_baseline_default['f1_non_emerg']*100:6.2f}%                  | {res_balanced_default['f1_non_emerg']*100:6.2f}%")
    print(f"{'Macro Average F1':<28} | {res_baseline_default['macro_f1']*100:6.2f}%                  | {res_balanced_default['macro_f1']*100:6.2f}%")
    print("-" * 82)
    print(f"{'True Negatives (TN)':<28} | {res_baseline_default['tn']:<24,d} | {res_balanced_default['tn']:<24,d}")
    print(f"{'False Positives (FP)':<28} | {res_baseline_default['fp']:<24,d} | {res_balanced_default['fp']:<24,d}")
    print(f"{'False Negatives (FN)':<28} | {res_baseline_default['fn']:<24,d} | {res_balanced_default['fn']:<24,d}")
    print(f"{'True Positives (TP)':<28} | {res_baseline_default['tp']:<24,d} | {res_balanced_default['tp']:<24,d}")

    print("\n" + "=" * 90)
    print(" THRESHOLD SWEEP TABLE: EXPERIMENT B (BALANCED MODEL)")
    print("=" * 90)
    print(" Threshold | Accuracy | Emerg Prec | Emerg Rec | Emerg F1 | Non-Em Prec | Non-Em Rec | Macro F1 |  FP  |  FN ")
    print("-" * 95)
    for r in sweep_balanced:
        print(
            f"   {r['threshold']:0.2f}    |  {r['accuracy']*100:5.2f}%  |   {r['prec_emerg']*100:5.2f}%   |  {r['rec_emerg']*100:5.2f}%  |  {r['f1_emerg']*100:5.2f}%  |"
            f"   {r['prec_non_emerg']*100:5.2f}%   |   {r['rec_non_emerg']*100:5.2f}%   |  {r['macro_f1']*100:5.2f}%  | {r['fp']:4d} | {r['fn']:4d}"
        )

    print("\n" + "=" * 90)
    print(" BENCHMARK TEST SAMPLES")
    print("=" * 90)
    for i, (b_item, bal_item) in enumerate(zip(preds_baseline_samples, preds_balanced_samples), 1):
        print(f"[{i}] \"{b_item['text']}\"")
        print(f"    Exp A (Baseline): {b_item['pred']:<18} | P(emerg)={b_item['p_emerg']*100:5.1f}%")
        print(f"    Exp B (Balanced): {bal_item['pred_0.5']:<18} | P(emerg)={bal_item['p_emerg']*100:5.1f}%")

    print("\n" + "=" * 90)
    print(" [OK] BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
