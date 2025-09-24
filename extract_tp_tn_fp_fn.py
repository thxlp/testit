import os
import pandas as pd
import numpy as np
from joblib import load


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    text_col = None
    label_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if text_col is None and ("text" in col_lower or "email" in col_lower):
            text_col = col
        if label_col is None and ("type" in col_lower or "label" in col_lower):
            label_col = col
    if text_col is None or label_col is None:
        raise ValueError("Cannot detect text/label columns. Expected names containing 'text'/'email' and 'type'/'label'.")
    return text_col, label_col


def main() -> None:
    print("📦 Extract TP/TN/FP/FN from test dataset")
    df = pd.read_csv("Phishing_Email_test.csv", encoding="utf-8")
    text_col, label_col = detect_columns(df)

    texts = df[text_col].astype(str).fillna("").tolist()
    labels = df[label_col].astype(str).tolist()

    # Load model and vectorizer
    model = load("best_logisticregression_model.joblib")
    vectorizer = load("best_logisticregression_vectorizer.joblib")

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None

    # Build label mapping from dataset labels to numeric indices used for evaluation
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    y_true = [label_to_idx[label] for label in labels]
    y_pred = preds.tolist()

    # Define positive class as 'Phishing Email' if present; otherwise use the first class
    if "Phishing Email" in label_to_idx:
        positive_idx = label_to_idx["Phishing Email"]
    else:
        positive_idx = 0

    rows = []
    for i in range(len(df)):
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        true_label = idx_to_label.get(true_idx, str(true_idx))
        pred_label = idx_to_label.get(pred_idx, str(pred_idx))
        confidence = None
        if proba is not None:
            try:
                confidence = float(np.max(proba[i]))
            except Exception:
                confidence = None

        if true_idx == positive_idx and pred_idx == positive_idx:
            tag = "TP"
        elif true_idx != positive_idx and pred_idx != positive_idx:
            tag = "TN"
        elif true_idx != positive_idx and pred_idx == positive_idx:
            tag = "FP"
        else:
            tag = "FN"

        rows.append({
            "index": i,
            text_col: texts[i],
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "tag": tag,
        })

    out_df = pd.DataFrame(rows)

    out_dir = os.path.join(os.getcwd(), "comparison_output")
    os.makedirs(out_dir, exist_ok=True)

    # Save combined
    out_df.to_csv(os.path.join(out_dir, "predictions_with_confidence.csv"), index=False, encoding="utf-8")

    # Save splits
    out_df[out_df["tag"] == "TP"].to_csv(os.path.join(out_dir, "true_positives.csv"), index=False, encoding="utf-8")
    out_df[out_df["tag"] == "TN"].to_csv(os.path.join(out_dir, "true_negatives.csv"), index=False, encoding="utf-8")
    out_df[out_df["tag"] == "FP"].to_csv(os.path.join(out_dir, "false_positives.csv"), index=False, encoding="utf-8")
    out_df[out_df["tag"] == "FN"].to_csv(os.path.join(out_dir, "false_negatives.csv"), index=False, encoding="utf-8")

    print(f"✅ Saved to: {out_dir}")


if __name__ == "__main__":
    main()


