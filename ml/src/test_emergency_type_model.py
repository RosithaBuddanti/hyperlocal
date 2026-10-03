"""
Inference & Testing Script for Multi-Label Emergency Type Classifier
Hyperlocal Emergency Response Platform - Machine Learning Workspace

Loads: ml/models/emergency_type_classifier.joblib
Evaluates sample emergency scenarios and displays predicted emergency types and probabilities.
"""

import sys
import warnings
from pathlib import Path
import joblib
import numpy as np

# Suppress optimizer warnings & configure UTF-8 output
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_model():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    model_path = project_root / "ml" / "models" / "emergency_type_classifier.joblib"
    labels_path = project_root / "ml" / "models" / "emergency_type_labels.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}. Run train_emergency_type_model.py first.")

    pipeline = joblib.load(model_path)

    if labels_path.exists():
        label_meta = joblib.load(labels_path)
        classes = label_meta.get("classes", [])
    elif hasattr(pipeline, "custom_label_names_"):
        classes = pipeline.custom_label_names_
    elif hasattr(pipeline, "classes_"):
        classes = list(pipeline.classes_)
    else:
        raise ValueError("Could not determine class labels from model or metadata.")

    return pipeline, classes


def test_emergency_type_model(custom_texts=None, threshold=0.5):
    print("=" * 80)
    print(" MULTI-LABEL EMERGENCY TYPE CLASSIFIER - INFERENCE TEST")
    print("=" * 80)

    pipeline, classes = load_model()
    print(f"Loaded model successfully. Number of classes: {len(classes)}")

    default_examples = [
        "There is a large fire near the highway and people need immediate help.",
        "Heavy flooding has trapped people inside several houses.",
        "A vehicle crashed and several people are injured.",
        "The earthquake damaged buildings and people need shelter.",
        "Electricity is down across the affected area.",
    ]

    test_samples = custom_texts if custom_texts else default_examples

    for i, text in enumerate(test_samples, 1):
        print(f"\n--- Example {i} ---")
        print("Input:")
        print(text)

        # Binary prediction (0 or 1 per class from classifier)
        pred_binary = pipeline.predict([text])[0]
        # Probability per class
        pred_probs = pipeline.predict_proba([text])[0]

        # Extract active labels
        predicted_types = [classes[idx] for idx, val in enumerate(pred_binary) if val == 1]

        # Sort all class probabilities descending
        class_prob_pairs = sorted(zip(classes, pred_probs), key=lambda x: x[1], reverse=True)

        print("\nPredicted emergency types:")
        if predicted_types:
            print(", ".join(predicted_types))
        else:
            print("None (no emergency types exceeded decision threshold)")

        print("\nPrediction probabilities:")
        # Display probabilities for predicted types first, followed by top overall classes
        print("  [Active Predictions]:")
        if predicted_types:
            for lbl in predicted_types:
                prob = pred_probs[classes.index(lbl)]
                print(f"    - {lbl:<25}: {prob * 100:6.2f}%")
        else:
            print("    (none)")

        print("  [Top 5 Most Confident Labels Overall]:")
        for lbl, prob in class_prob_pairs[:5]:
            flag = " [ACTIVE]" if lbl in predicted_types else ""
            print(f"    - {lbl:<25}: {prob * 100:6.2f}%{flag}")

    print("\n" + "=" * 80)
    print("Inference test complete.")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided text on CLI
        user_input = " ".join(sys.argv[1:])
        test_emergency_type_model([user_input])
    else:
        test_emergency_type_model()
