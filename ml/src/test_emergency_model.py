"""
Emergency Detection Baseline Model Tester
Hyperlocal Emergency Response Platform - Machine Learning Workspace

Loads ml/models/emergency_classifier.joblib and evaluates inference on sample text inputs.
"""

import os
import sys
from pathlib import Path
import joblib

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_emergency_model():
    print("=" * 80)
    print(" TESTING BINARY EMERGENCY DETECTION MODEL")
    print("=" * 80)

    # 1. Resolve model path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    model_path = project_root / "ml" / "models" / "emergency_classifier.joblib"

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please run 'ml/src/train_emergency_model.py' first.")
        sys.exit(1)

    print(f"\n[Step 1] Loading model pipeline from:\n  {model_path}")
    pipeline = joblib.load(model_path)
    print("Model loaded successfully.")

    # 2. Define test messages
    test_messages = [
        "There is a major fire near the highway and people need help.",
        "I lost my wallet yesterday.",
        "A car has crashed and someone appears injured.",
        "There is heavy flooding and people are trapped.",
        "Can someone tell me today's weather?",
    ]

    # 3. Perform inference
    print("\n[Step 2] Running inference on benchmark test messages:\n")
    print("-" * 80)

    for idx, message in enumerate(test_messages, start=1):
        prediction = pipeline.predict([message])[0]
        probabilities = pipeline.predict_proba([message])[0]
        
        # Classes: 0 = non-emergency, 1 = emergency
        pred_label = "emergency (1)" if prediction == 1 else "non-emergency (0)"
        confidence = probabilities[prediction] * 100
        prob_non_emerg = probabilities[0] * 100
        prob_emerg = probabilities[1] * 100

        print(f"Example #{idx}:")
        print(f"  Text                     : \"{message}\"")
        print(f"  Predicted Emergency Class: {pred_label}")
        print(f"  Confidence Score         : {confidence:.2f}%")
        print(f"  Class Probabilities      : Non-Emergency: {prob_non_emerg:.2f}% | Emergency: {prob_emerg:.2f}%")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(" [OK] EMERGENCY MODEL TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_emergency_model()
