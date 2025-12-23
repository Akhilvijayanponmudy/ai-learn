import json
from pathlib import Path
import joblib

MODEL_PATH = Path("data/generated/classifier_model.joblib")
METRICS_PATH = Path("data/generated/classifier_metrics.json")


def load_label_map():
    """
    Best effort:
    1) If classifier_metrics.json exists and contains "classes", use that order.
    2) Otherwise, fallback to a sensible default for 4-class intent.
    """
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            classes = metrics.get("classes")
            if isinstance(classes, list) and classes:
                # If classes are already strings, map string->string
                if isinstance(classes[0], str):
                    return {c: c for c in classes}
                # If classes are ints, we still need names; but metrics usually stores strings.
        except Exception:
            pass

    # Fallback (edit if your label ids differ)
    return {
        0: "billing",
        1: "bug",
        2: "feature_request",
        3: "how-to",
        "0": "billing",
        "1": "bug",
        "2": "feature_request",
        "3": "how-to",
    }


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}. Run scripts/01_train_eval_classifier.py first.")

    pipe = joblib.load(MODEL_PATH)
    label_map = load_label_map()

    # Pipeline classes (may be ints or strings depending on training labels)
    classes = getattr(pipe, "classes_", None)
    if classes is None:
        raise SystemExit("Loaded model does not expose classes_. Something is wrong with the saved pipeline.")

    print("Classifier loaded.")
    print(f"Classes: {list(classes)}")

    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        probs = pipe.predict_proba([q])[0]
        pred = pipe.predict([q])[0]

        print("\nPrediction probabilities:")
        for cls, p in zip(classes, probs):
            # Convert cls to something printable, then map to friendly name if possible
            friendly = label_map.get(cls, label_map.get(str(cls), str(cls)))
            print(f"  {str(friendly):15s}: {float(p):.3f}")

        final_friendly = label_map.get(pred, label_map.get(str(pred), str(pred)))
        print(f"\nFinal label: {final_friendly}")


if __name__ == "__main__":
    main()
