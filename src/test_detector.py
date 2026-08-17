"""
Interactive AI-Text Detector Tester
Run this locally where your models/ folder with the .pkl files is located.

Usage:
    python test_detector.py
Then type any text and press Enter to get a prediction.
Type 'quit' or 'exit' to stop.
"""

import pickle
import os

MODEL_PATH = "models/baseline_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(f"ERROR: Could not find model files.")
        print(f"Expected: {MODEL_PATH} and {VECTORIZER_PATH}")
        print("Make sure you run this script from the folder containing your 'models/' directory,")
        print("or edit MODEL_PATH / VECTORIZER_PATH at the top of this file.")
        exit(1)

    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    return model, vectorizer

def predict_text(text, model, vectorizer):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]

    # Get confidence if the model supports predict_proba
    confidence_str = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf_dict = dict(zip(model.classes_, proba))
        confidence_str = " | Confidence: " + ", ".join(
            f"{cls}: {p*100:.1f}%" for cls, p in conf_dict.items()
        )

    return pred, confidence_str

def main():
    print("=" * 60)
    print("  AI-GENERATED TEXT DETECTOR — Interactive Tester")
    print("=" * 60)
    print("Loading model...")
    model, vectorizer = load_model()
    print("Model loaded successfully!\n")
    print("Type or paste any text below and press Enter to check it.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        text = input(">> Enter text: ").strip()

        if text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not text:
            print("Please enter some text.\n")
            continue

        prediction, confidence = predict_text(text, model, vectorizer)
        label = "🤖 AI-GENERATED" if prediction == "ai" else "🧑 HUMAN-WRITTEN"

        print(f"\n   Result: {label}{confidence}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()