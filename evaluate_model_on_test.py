"""
Evaluate saved model and vectorizer on the test CSV and print per-class percentages.

Usage:
python .\evaluate_model_on_test.py --test Phishing_Email_test.csv --model phishing_detector_model.pkl --vectorizer tfidf_vectorizer.pkl --outdir .\comparison_output

Outputs:
- prints accuracy and per-class precision/recall/f1 and confusion matrix percentages
- writes JSON report to outdir/evaluation_report.json
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from typing import List

try:
    import pandas as pd
except Exception:
    pd = None

import csv

# Increase CSV field size limit
import sys
try:
    csv.field_size_limit(sys.maxsize)
except Exception:
    try:
        csv.field_size_limit(2 ** 31 - 1)
    except Exception:
        pass

try:
    from joblib import load as joblib_load
except Exception:
    import pickle
    joblib_load = None

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np


def open_with_encoding(path: str):
    return open(path, 'r', encoding='utf-8', errors='replace')


def get_best_text_label_cols(columns: List[str]):
    lower = [c.lower() for c in columns]
    text_cols = ['email text', 'text', 'email_text', 'text_combined', 'body', 'message', 'content']
    label_cols = ['email type', 'email_type', 'label', 'type', 'email type']
    text = None
    label = None
    for cand in text_cols:
        for i, c in enumerate(lower):
            if c == cand or cand in c:
                text = columns[i]
                break
        if text:
            break
    for cand in label_cols:
        for i, c in enumerate(lower):
            if c == cand or cand in c:
                label = columns[i]
                break
        if label:
            break
    return text, label


def load_test_df(path: str, text_col: str = None, label_col: str = None):
    if pd is not None:
        df = pd.read_csv(path, encoding='utf-8', engine='python')
        cols = [c for c in df.columns.tolist()]
        if text_col is None or label_col is None:
            t, l = get_best_text_label_cols(cols)
            if text_col is None:
                text_col = t
            if label_col is None:
                label_col = l
        if text_col is None or label_col is None:
            raise RuntimeError('Could not infer text/label columns; pass them explicitly')
        df = df[[text_col, label_col]].rename(columns={text_col: 'email_text', label_col: 'label'})
        return df
    # fallback to csv
    with open_with_encoding(path) as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError('Empty CSV')
        header = [h.strip() for h in header]
        if text_col is None or label_col is None:
            t, l = get_best_text_label_cols(header)
            if text_col is None:
                text_col = t
            if label_col is None:
                label_col = l
        if text_col is None or label_col is None:
            raise RuntimeError('Could not infer text/label columns from header')
        text_idx = header.index(text_col)
        label_idx = header.index(label_col)
        rows = []
        for row in reader:
            rows.append({'email_text': row[text_idx] if text_idx < len(row) else '', 'label': row[label_idx] if label_idx < len(row) else ''})
        # lightweight DataFrame-like structure
        return rows


def load_pickle(path: str):
    if joblib_load is not None:
        return joblib_load(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True)
    parser.add_argument('--model', default='phishing_detector_model.pkl')
    parser.add_argument('--vectorizer', default='tfidf_vectorizer.pkl')
    parser.add_argument('--outdir', default='./comparison_output')
    parser.add_argument('--examples', type=int, default=20, help='number of false pos/neg examples to save')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load test data
    df = load_test_df(args.test)
    # ensure pandas-like
    if pd is not None:
        texts = df['email_text'].astype(str).tolist()
        true_labels = df['label'].astype(str).tolist()
    else:
        texts = [r['email_text'] for r in df]
        true_labels = [r['label'] for r in df]

    # load vectorizer and model
    vec = load_pickle(args.vectorizer)
    model = load_pickle(args.model)

    X = vec.transform(texts)
    preds = model.predict(X)

    # normalize to strings to avoid dtype mismatches (nullable ints vs strings)
    true_labels = [str(x) for x in true_labels]
    preds = [str(x) for x in preds]

    # Compute prediction probabilities / confidences
    probs = None
    classes_from_model = None
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            classes_from_model = [str(c) for c in model.classes_]
        else:
            # fallback to decision_function
            dfun = model.decision_function(X)
            if dfun.ndim == 1:
                # binary: convert to probability with sigmoid
                from math import exp
                pos_probs = [1.0 / (1.0 + exp(-float(v))) for v in dfun]
                probs = np.vstack([1.0 - np.array(pos_probs), np.array(pos_probs)]).T
                classes_from_model = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else ['0', '1']
            else:
                # multiclass: softmax
                exp_scores = np.exp(dfun - np.max(dfun, axis=1, keepdims=True))
                probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                classes_from_model = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else [str(i) for i in range(probs.shape[1])]
    except Exception:
        probs = None
        classes_from_model = [str(c) for c in getattr(model, 'classes_', [])]

    # ensure preds are strings
    preds = [str(p) for p in preds]

    # If model predicted numeric labels but test labels are descriptive, attempt mapping
    unique_true = set(true_labels)
    unique_preds = set(preds)
    descriptive = {'Phishing Email', 'Safe Email'}
    if unique_preds <= {'0', '1'} and (descriptive & unique_true):
        num_to_desc = {'1': 'Phishing Email', '0': 'Safe Email'}
        preds = [num_to_desc.get(p, p) for p in preds]
        if probs is not None and classes_from_model is not None:
            # reorder/rename prob columns to match mapped labels
            # build mapping from index to new label
            col_labels = [str(c) for c in classes_from_model]
            new_cols = [num_to_desc.get(c, c) for c in col_labels]
            classes_from_model = new_cols

    # If model outputs numeric labels ("0","1") but true labels are descriptive
    # like 'Phishing Email'/'Safe Email', map numeric preds to those descriptive
    unique_true = set(true_labels)
    unique_preds = set(preds)
    # common descriptive labels we'd expect in the test set
    descriptive = {'Phishing Email', 'Safe Email'}
    if unique_preds <= {'0', '1'} and (descriptive & unique_true):
        # build mapping: prefer test descriptive labels if present
        if 'Phishing Email' in unique_true and 'Safe Email' in unique_true:
            num_to_desc = {'1': 'Phishing Email', '0': 'Safe Email'}
        else:
            # fallback mapping: assume '1' -> phishing
            num_to_desc = {'1': next(iter(descriptive)) if descriptive & unique_true else 'Phishing Email', '0': 'Safe Email'}
        preds = [num_to_desc.get(p, p) for p in preds]

    acc = accuracy_score(true_labels, preds)
    labels = sorted(list(set(true_labels) | set(preds)))
    prec, rec, f1, sup = precision_recall_fscore_support(true_labels, preds, labels=labels, zero_division=0)
    cm = confusion_matrix(true_labels, preds, labels=labels)

    # convert confusion matrix to percentages per true-label row
    cm_percent = []
    for i, row in enumerate(cm):
        total = row.sum()
        if total == 0:
            cm_percent.append([0.0 for _ in row])
        else:
            cm_percent.append([float(x) / float(total) for x in row])

    report = {
        'accuracy': float(acc),
        'labels': labels,
        'per_label': {labels[i]: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(sup[i])} for i in range(len(labels))},
        'confusion_matrix_counts': cm.tolist(),
        'confusion_matrix_percent': cm_percent
    }

    # Per-sample confidence and per-class average confidence
    sample_confidences = []
    per_class_conf = {}
    if probs is not None and classes_from_model is not None:
        # map classes_from_model (list) to column index
        for i, p in enumerate(preds):
            # find predicted label index in classes_from_model
            try:
                idx = classes_from_model.index(p)
                conf = float(probs[i, idx])
            except Exception:
                # fallback: take max prob
                conf = float(max(probs[i].tolist()))
            sample_confidences.append(conf)
        # aggregate per true label
        conf_by_true = {}
        for t, c in zip(true_labels, sample_confidences):
            conf_by_true.setdefault(t, []).append(c)
        for k, v in conf_by_true.items():
            per_class_conf[k] = float(sum(v) / len(v)) if v else 0.0
    else:
        # no probability available; default confidences to None
        sample_confidences = [None] * len(preds)

    report['per_class_average_confidence'] = per_class_conf

    # compute probability assigned to true label per sample (if probs available)
    prob_true = [None] * len(texts)
    if probs is not None and classes_from_model is not None:
        for i, t in enumerate(true_labels):
            try:
                idx = classes_from_model.index(t)
                prob_true[i] = float(probs[i, idx])
            except Exception:
                prob_true[i] = None

    # per-class confidence statistics (mean, median, p25, p75, std, count) on prob_true
    per_class_stats = {}
    if any(p is not None for p in prob_true):
        from statistics import mean, median, pstdev
        import numpy as _np
        vals_by_class = {}
        for t, v in zip(true_labels, prob_true):
            if v is None:
                continue
            vals_by_class.setdefault(t, []).append(v)
        for k, vals in vals_by_class.items():
            arr = _np.array(vals)
            per_class_stats[k] = {
                'mean': float(arr.mean()),
                'median': float(_np.median(arr)),
                'p25': float(_np.percentile(arr, 25)),
                'p75': float(_np.percentile(arr, 75)),
                'std': float(arr.std(ddof=0)),
                'count': int(arr.size)
            }
    report['per_class_confidence_stats'] = per_class_stats

    # Save per-sample CSV with predictions and confidence (small memory footprint write)
    pred_csv = os.path.join(args.outdir, 'predictions_with_confidence.csv')
    with open(pred_csv, 'w', encoding='utf-8', newline='') as outf:
        import csv as _csv
        w = _csv.writer(outf)
        w.writerow(['email_text', 'true_label', 'predicted_label', 'confidence'])
        for t, tr, pr, cf in zip(texts, true_labels, preds, sample_confidences):
            w.writerow([t[:1000], tr, pr, '' if cf is None else f"{cf:.6f}"])
    report['predictions_with_confidence_csv'] = pred_csv

    # Save false positives and false negatives examples
    fp_path = os.path.join(args.outdir, 'false_positives.csv')
    fn_path = os.path.join(args.outdir, 'false_negatives.csv')
    fp_rows = []
    fn_rows = []
    for t_text, t_true, t_pred, p_pred, p_true_val in zip(texts, true_labels, preds, sample_confidences, prob_true):
        if t_true != t_pred and t_pred == 'Phishing Email' and t_true == 'Safe Email':
            fp_rows.append((t_text, t_true, t_pred, p_pred, p_true_val))
        if t_true != t_pred and t_true == 'Phishing Email' and t_pred == 'Safe Email':
            fn_rows.append((t_text, t_true, t_pred, p_pred, p_true_val))

    # sort and keep top N examples
    N = max(0, int(args.examples))
    # false positives: highest predicted phishing confidence first
    fp_rows_sorted = sorted([r for r in fp_rows if r[3] is not None], key=lambda x: (-(x[3] if x[3] is not None else 0)))
    fn_rows_sorted = sorted([r for r in fn_rows if r[4] is not None], key=lambda x: ((x[4] if x[4] is not None else 1)))

    with open(fp_path, 'w', encoding='utf-8', newline='') as outf:
        import csv as _csv
        w = _csv.writer(outf)
        w.writerow(['email_text', 'true_label', 'predicted_label', 'predicted_label_confidence', 'prob_true_label'])
        for row in fp_rows_sorted[:N]:
            t_text = row[0][:1000]
            w.writerow([t_text, row[1], row[2], '' if row[3] is None else f"{row[3]:.6f}", '' if row[4] is None else f"{row[4]:.6f}"])

    with open(fn_path, 'w', encoding='utf-8', newline='') as outf:
        import csv as _csv
        w = _csv.writer(outf)
        w.writerow(['email_text', 'true_label', 'predicted_label', 'predicted_label_confidence', 'prob_true_label'])
        for row in fn_rows_sorted[:N]:
            t_text = row[0][:1000]
            w.writerow([t_text, row[1], row[2], '' if row[3] is None else f"{row[3]:.6f}", '' if row[4] is None else f"{row[4]:.6f}"])

    report['false_positives_csv'] = fp_path
    report['false_negatives_csv'] = fn_path

    outpath = os.path.join(args.outdir, 'evaluation_report.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print concise summary with percentages
    print('Accuracy: {:.2f}%'.format(report['accuracy'] * 100))
    print('\nPer-label results:')
    for lab, stats in report['per_label'].items():
        print(f"- {lab}: precision={stats['precision']*100:.2f}%, recall={stats['recall']*100:.2f}%, f1={stats['f1']*100:.2f}%, support={stats['support']}")

    print('\nConfusion matrix (percent per true label row):')
    header = [''] + labels
    print('\t'.join(header))
    for i, lab in enumerate(labels):
        rowp = '\t'.join([f"{p*100:.1f}%" for p in report['confusion_matrix_percent'][i]])
        print(f"{lab}\t{rowp}")

    # print per-class average confidence if present
    if report.get('per_class_average_confidence'):
        print('\nPer-class average confidence:')
        for k, v in report['per_class_average_confidence'].items():
            print(f"- {k}: {v*100:.2f}%")

    print('\nPer-sample predictions CSV:', report.get('predictions_with_confidence_csv'))
    print('Report written to', outpath)


if __name__ == '__main__':
    main()
