"""
Dataset Preparation Script for Hyperlocal Emergency Response Platform
Prepares clean intermediate train/test CSVs from 'hotal/emergency_classification'.
"""

import ast
import os
import sys
from collections import Counter
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def prepare_dataset():
    print("=" * 80)
    print(" PREPARING CLEAN DATASET: hotal/emergency_classification")
    print("=" * 80)

    # 1. Resolve target directories
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    processed_dir = project_root / "ml" / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_out_path = processed_dir / "train.csv"
    test_out_path = processed_dir / "test.csv"
    summary_out_path = processed_dir / "dataset_summary.txt"

    # 2. Load dataset from cache / Hub
    print("\n[Step 1] Loading dataset from local cache / Hugging Face...")
    ds = load_dataset("hotal/emergency_classification")

    raw_train_df = ds["train"].to_pandas()
    raw_test_df = ds["test"].to_pandas()

    raw_train_count = len(raw_train_df)
    raw_test_count = len(raw_test_df)
    print(f"Loaded raw splits: train={raw_train_count:,} rows, test={raw_test_count:,} rows")

    # 3. Clean each split
    def clean_split(df: pd.DataFrame, split_name: str):
        initial_count = len(df)
        
        # Rename 'message' to 'text' as primary text column
        df = df.rename(columns={"message": "text"}).copy()

        # Remove Unnamed: 0, raw classes, and unnecessary metadata columns (original, genre)
        cols_to_remove = ["Unnamed: 0", "classes", "original", "genre"]
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

        # Strip whitespace from text
        df["text"] = df["text"].astype(str).str.strip()

        # Identify and remove empty or null messages
        empty_mask = (df["text"].isna()) | (df["text"] == "") | (df["text"].str.lower() == "nan")
        empty_count = int(empty_mask.sum())
        df = df[~empty_mask].copy()

        # Identify and remove exact duplicate messages within the split (preserving first occurrence)
        dup_mask = df.duplicated(subset=["text"], keep="first")
        dup_count = int(dup_mask.sum())
        df = df[~dup_mask].copy()

        # Ensure primary columns come first
        primary_cols = ["text", "emergency", "class_labels"]
        other_cols = [c for c in df.columns if c not in primary_cols]
        ordered_cols = primary_cols + other_cols
        df = df[ordered_cols].copy()

        print(f"  Split '{split_name}':")
        print(f"    - Initial rows: {initial_count:,}")
        print(f"    - Empty/null messages removed: {empty_count:,}")
        print(f"    - Duplicate messages removed: {dup_count:,}")
        print(f"    - Final clean rows: {len(df):,}")

        return df, empty_count, dup_count

    print("\n[Step 2] Cleaning and filtering data...")
    clean_train_df, train_empty_removed, train_dup_removed = clean_split(raw_train_df, "train")
    clean_test_df, test_empty_removed, test_dup_removed = clean_split(raw_test_df, "test")

    # 4. Save processed CSV files
    print("\n[Step 3] Saving processed CSV files...")
    clean_train_df.to_csv(train_out_path, index=False, encoding="utf-8")
    clean_test_df.to_csv(test_out_path, index=False, encoding="utf-8")
    print(f"  -> Saved train dataset to: {train_out_path}")
    print(f"  -> Saved test dataset to:  {test_out_path}")

    # 5. Label frequency and distributions
    total_clean_records = len(clean_train_df) + len(clean_test_df)
    combined_class_labels = list(clean_train_df["class_labels"]) + list(clean_test_df["class_labels"])

    all_labels = []
    for cl in combined_class_labels:
        try:
            labels = ast.literal_eval(cl)
            all_labels.extend(labels)
        except Exception:
            pass

    label_counter = Counter(all_labels)

    # 6. Generate dataset_summary.txt
    print("\n[Step 4] Generating summary report...")
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(" HYPERLOCAL EMERGENCY RESPONSE PLATFORM - ML DATASET SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("1. RECORD COUNTS:")
    summary_lines.append(f"   - Total Clean Records  : {total_clean_records:,}")
    summary_lines.append(f"   - Clean Train Records  : {len(clean_train_df):,}")
    summary_lines.append(f"   - Clean Test Records   : {len(clean_test_df):,}")
    summary_lines.append(f"   - Train Split Ratio    : {(len(clean_train_df) / total_clean_records) * 100:.2f}%")
    summary_lines.append(f"   - Test Split Ratio     : {(len(clean_test_df) / total_clean_records) * 100:.2f}%")
    summary_lines.append("")
    summary_lines.append("2. DATA CLEANING & DEDUPLICATION:")
    summary_lines.append(f"   - Raw Train Rows       : {raw_train_count:,}")
    summary_lines.append(f"   - Raw Test Rows        : {raw_test_count:,}")
    summary_lines.append(f"   - Empty Messages Removed (Train): {train_empty_removed:,}")
    summary_lines.append(f"   - Empty Messages Removed (Test) : {test_empty_removed:,}")
    summary_lines.append(f"   - Duplicate Messages Removed (Train): {train_dup_removed:,}")
    summary_lines.append(f"   - Duplicate Messages Removed (Test) : {test_dup_removed:,}")
    summary_lines.append(f"   - Total Removed Records: {train_empty_removed + test_empty_removed + train_dup_removed + test_dup_removed:,}")
    summary_lines.append("")
    summary_lines.append("3. DATASET COLUMNS & SCHEMA:")
    summary_lines.append(f"   - Total Columns: {len(clean_train_df.columns)}")
    summary_lines.append(f"   - Columns: {list(clean_train_df.columns)}")
    summary_lines.append("   - Primary Features:")
    summary_lines.append("     * 'text'         : Cleaned emergency distress message in English")
    summary_lines.append("     * 'emergency'    : Binary indicator (1 = Emergency, 0 = Non-emergency)")
    summary_lines.append("     * 'class_labels' : List of active disaster / hazard labels")
    summary_lines.append("     * 36 Category Columns: Binary one-hot flags for each disaster category")
    summary_lines.append("")
    summary_lines.append("4. LABEL DISTRIBUTION & FREQUENCY (Combined Clean Dataset):")
    summary_lines.append(f"   - Number of Unique Labels: {len(label_counter)}")
    summary_lines.append("   " + "-" * 70)
    summary_lines.append(f"   {'Rank':<5} {'Label Name':<26} {'Occurrences':<14} {'Message %':<10}")
    summary_lines.append("   " + "-" * 70)
    for rank, (label, count) in enumerate(label_counter.most_common(), start=1):
        pct = (count / total_clean_records) * 100
        summary_lines.append(f"   {rank:<5} {label:<26} {count:<14,d} {pct:6.2f}%")
    summary_lines.append("   " + "-" * 70)
    summary_lines.append("")
    summary_lines.append("5. FIRST 5 CLEANED RECORDS (Train Split Sample):")
    for idx in range(min(5, len(clean_train_df))):
        rec = clean_train_df.iloc[idx]
        summary_lines.append(f"\n   [Record #{idx + 1}]")
        summary_lines.append(f"   Text        : {rec['text']}")
        summary_lines.append(f"   Emergency   : {rec['emergency']}")
        summary_lines.append(f"   Class Labels: {rec['class_labels']}")
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append(" END OF DATASET SUMMARY")
    summary_lines.append("=" * 80)

    summary_text = "\n".join(summary_lines)
    with open(summary_out_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"  -> Saved dataset summary to: {summary_out_path}")

    # 7. Print summary highlights to console
    print("\n" + "=" * 80)
    print(" SUMMARY HIGHLIGHTS")
    print("=" * 80)
    print(f"Total Records Processed : {total_clean_records:,}")
    print(f"Train Records           : {len(clean_train_df):,}")
    print(f"Test Records            : {len(clean_test_df):,}")
    print(f"Empty Records Removed   : {train_empty_removed + test_empty_removed:,}")
    print(f"Duplicate Records Removed: {train_dup_removed + test_dup_removed:,}")
    print(f"Unique Labels Found     : {len(label_counter)}")

    print("\nFirst 5 Cleaned Train Records:")
    for i in range(min(5, len(clean_train_df))):
        r = clean_train_df.iloc[i]
        print(f"  [{i+1}] text='{r['text'][:65]}...' | emergency={r['emergency']} | class_labels={r['class_labels']}")

    print("\n" + "=" * 80)
    print(" [OK] DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    prepare_dataset()
