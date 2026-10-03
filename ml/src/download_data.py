import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def main():
    print("Loading dataset 'hotal/emergency_classification' from Hugging Face...")
    dataset = load_dataset("hotal/emergency_classification")

    print("\n================ DATASET STRUCTURE ================")
    print(dataset)

    # Determine train split or available splits
    print("\n================ SPLITS & ROWS ================")
    for split_name in dataset.keys():
        print(f"Split: '{split_name}', Number of rows: {len(dataset[split_name])}")

    # Primary split to use
    if "train" in dataset:
        target_split = dataset["train"]
        split_used = "train"
    else:
        split_used = list(dataset.keys())[0]
        target_split = dataset[split_used]

    columns = target_split.column_names
    print(f"\n================ AVAILABLE COLUMNS (Split: '{split_used}') ================")
    print(f"Total Columns ({len(columns)}): {columns}")

    print(f"\n================ FIRST 3 RECORDS (Split: '{split_used}') ================")
    for i in range(min(3, len(target_split))):
        print(f"\n--- Record {i + 1} ---")
        for col in columns:
            print(f"  {col}: {target_split[i][col]}")

    # Resolve paths relative to this script: ml/src/download_data.py -> ml/data/raw/emergency_classification.csv
    script_dir = Path(__file__).resolve().parent
    raw_data_dir = script_dir.parent / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    output_csv = raw_data_dir / "emergency_classification.csv"

    print(f"\nSaving training data to: {output_csv} ...")
    df = target_split.to_pandas()
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Done! Successfully saved {len(df)} rows to: {output_csv}")

if __name__ == "__main__":
    main()
