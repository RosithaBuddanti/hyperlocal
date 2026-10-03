"""
Dataset Inspector for 'hotal/emergency_classification'
Hyperlocal Emergency Response Platform - Machine Learning Workspace
"""

import ast
import sys
from collections import Counter
from datasets import load_dataset

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def inspect_dataset():
    print("=" * 80)
    print(" DATASET INSPECTOR: hotal/emergency_classification")
    print("=" * 80)

    # 1. Load dataset
    print("\n[STEP 1] Loading dataset from Hugging Face...")
    dataset = load_dataset("hotal/emergency_classification")

    # 2. Print dataset structure
    print("\n[STEP 2] Dataset Structure:")
    print(dataset)

    # 3. Print number of records
    print("\n[STEP 3] Number of Records:")
    total_records = 0
    for split_name, split_data in dataset.items():
        print(f"  - Split '{split_name}': {len(split_data):,} rows")
        total_records += len(split_data)
    print(f"  - Total across all splits: {total_records:,} rows")

    train_data = dataset["train"]

    # 4. Print column names & data types
    print(f"\n[STEP 4] Available Columns ({len(train_data.column_names)} columns):")
    for i, col in enumerate(train_data.column_names, start=1):
        dtype = train_data.features[col].dtype
        print(f"  {i:2d}. {col:25s} (type: {dtype})")

    # 5. Print first 3 records
    print("\n[STEP 5] First 3 Records from Train Split:")
    for idx in range(min(3, len(train_data))):
        record = train_data[idx]
        print(f"\n--- Record #{idx + 1} ---")
        print(f"  Index (Unnamed: 0) : {record.get('Unnamed: 0')}")
        print(f"  Message            : {record.get('message')}")
        print(f"  Original           : {record.get('original')}")
        print(f"  Genre              : {record.get('genre')}")
        print(f"  Emergency Flag     : {record.get('emergency')}")
        print(f"  Class Labels       : {record.get('class_labels')}")
        print(f"  Raw Binary Classes : {record.get('classes')}")

    # 6. Identify the label column(s)
    binary_label_cols = [
        "emergency", "request", "offer", "aid_related", "medical_help",
        "medical_products", "search_and_rescue", "security", "military",
        "child_alone", "water", "food", "shelter", "clothing", "money",
        "missing_people", "refugees", "death", "other_aid",
        "infrastructure_related", "transport", "buildings", "electricity",
        "tools", "hospitals", "shops", "aid_centers", "other_infrastructure",
        "weather_related", "floods", "storm", "fire", "earthquake", "cold",
        "other_weather", "direct_report"
    ]

    print("\n[STEP 6] Identified Label Columns:")
    print("  1. 'class_labels' : Multi-label string list of active categories (e.g. \"['emergency', 'aid_related', 'shelter']\")")
    print("  2. 'classes'      : Raw string representing 36 binary flags (e.g. '[0, 0, 1, 0, ...]')")
    print(f"  3. 36 Binary Columns: Individual binary indicators (0 or 1) for each emergency disaster category.")

    # 7. Display unique labels and counts
    print("\n[STEP 7] Unique Labels & Frequencies in Train Split (20,791 records):")
    
    # Extract unique labels from 'class_labels'
    all_extracted_labels = []
    for label_str in train_data["class_labels"]:
        try:
            labels = ast.literal_eval(label_str)
            all_extracted_labels.extend(labels)
        except Exception:
            pass

    label_counts = Counter(all_extracted_labels)
    print(f"\n  A. Individual Classes Extracted from 'class_labels' ({len(label_counts)} unique active categories):")
    print(f"  {'Rank':<5} {'Category Label':<25} {'Count':<10} {'Percentage of Messages':<22}")
    print("  " + "-" * 65)
    for rank, (label, count) in enumerate(label_counts.most_common(), start=1):
        pct = (count / len(train_data)) * 100
        print(f"  {rank:<5} {label:<25} {count:<10,d} {pct:6.2f}%")

    print("\n  B. Summary of All 36 Binary Category Indicators:")
    print(f"  {'Category':<25} {'Positives (1)':<15} {'Negatives (0)':<15}")
    print("  " + "-" * 57)
    for col in binary_label_cols:
        pos = sum(1 for val in train_data[col] if val == 1)
        neg = len(train_data) - pos
        print(f"  {col:<25} {pos:<15,d} {neg:<15,d}")

    print("\n" + "=" * 80)
    print(" [OK] INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    inspect_dataset()
