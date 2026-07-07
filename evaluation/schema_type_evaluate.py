import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

EVAL_DIR = os.path.dirname(__file__)
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "..", "data", "validate_schema.csv")

RESULT_FILES = {
    "llama_8b_zero_shot":   "results_llama_8b_zero_shot.json",
    "llama_8b_few_shot":    "results_llama_8b_few_shot.json",
    "llama_70b_zero_shot":  "results_llama_70b_zero_shot.json",
    "llama_70b_few_shot":   "results_llama_70b_few_shot.json",
    "qwen_32b_zero_shot":   "results_qwen_32b_zero_shot.json",
    "qwen_32b_few_shot":    "results_qwen_32b_few_shot.json",
}

SCHEMA_TYPES = ["backend", "frontend", "database", "performance", "other"]


def load_ground_truth() -> Dict[str, str]:
    """bug_id -> human-labeled true schema_type (validate_schema.csv 'my_review' column)."""
    truth = {}
    with open(GROUND_TRUTH_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[row["bug_id"].strip()] = row["my_review"].strip().lower()
    return truth


def precision_recall_f1(rows: List[Dict], label: str):
    tp = sum(1 for r in rows if r["pred"] == label and r["true"] == label)
    fp = sum(1 for r in rows if r["pred"] == label and r["true"] != label)
    fn = sum(1 for r in rows if r["pred"] != label and r["true"] == label)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    support = tp + fn
    return p, r, f1, support


def confusion_matrix(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    labels = SCHEMA_TYPES + ["other_pred"]
    matrix = {t: {p: 0 for p in labels} for t in SCHEMA_TYPES}
    for r in rows:
        true, pred = r["true"], r["pred"]
        if true not in matrix:
            continue
        if pred in matrix[true]:
            matrix[true][pred] += 1
        else:
            matrix[true]["other_pred"] += 1
    return matrix


def print_confusion_matrix(matrix):
    cols = SCHEMA_TYPES + ["other_pred"]
    header = f"{'True \\ Pred':<14}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * (14 + 12 * len(cols)))
    for true_class in SCHEMA_TYPES:
        row = f"{true_class:<14}"
        for pred_class in cols:
            row += f"{matrix[true_class].get(pred_class, 0):>12}"
        print(row)


def evaluate_experiment(name: str, results: List[Dict], truth: Dict[str, str]) -> Dict:
    rows = []
    for r in results:
        bug_id = str(r.get("bug_id", "")).strip()
        if bug_id not in truth:
            continue
        rows.append({
            "bug_id": bug_id,
            "true": truth[bug_id],
            "pred": str(r.get("schema_type", "")).strip().lower(),
            "json_valid": r.get("json_valid", False),
        })

    if not rows:
        print(f"  ⚠ {name}: no overlapping bug_ids with ground truth — skipping")
        return None

    correct = sum(1 for r in rows if r["pred"] == r["true"])
    accuracy = correct / len(rows)

    f1s = [precision_recall_f1(rows, c)[2] for c in SCHEMA_TYPES]
    macro_f1 = sum(f1s) / len(f1s)

    total = len(rows)
    weighted_f1 = sum(
        precision_recall_f1(rows, c)[2] * precision_recall_f1(rows, c)[3] for c in SCHEMA_TYPES
    ) / total

    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    print(f"  Matched samples (in ground truth): {len(rows)}")
    print(f"  Schema-type accuracy : {accuracy*100:.1f}%")
    print(f"  Macro F1             : {macro_f1*100:.1f}%")
    print(f"  Weighted F1          : {weighted_f1*100:.1f}%")

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    per_class = {}
    for c in SCHEMA_TYPES:
        p, r, f1, support = precision_recall_f1(rows, c)
        print(f"  {c:<12} {p*100:>9.1f}% {r*100:>9.1f}% {f1*100:>9.1f}% {support:>10}")
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": support}

    print(f"\n  Confusion matrix:")
    print_confusion_matrix(confusion_matrix(rows))

    return {
        "name": name,
        "n": len(rows),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
    }


def run_full_schema_evaluation():
    truth = load_ground_truth()
    print(f"Loaded {len(truth)} ground-truth labels from {os.path.relpath(GROUND_TRUTH_PATH)}")

    summaries = []
    for name, fname in RESULT_FILES.items():
        path = os.path.join(EVAL_DIR, fname)
        if not os.path.exists(path):
            print(f"  ⚠ Missing: {path} — skipping")
            continue
        with open(path) as f:
            results = json.load(f)
        summary = evaluate_experiment(name, results, truth)
        if summary:
            summaries.append(summary)

    print(f"\n\n{'═'*80}")
    print("  SCHEMA TYPE — COMPARATIVE SUMMARY")
    print(f"{'═'*80}")
    print(f"  {'Experiment':<22} {'N':>5} {'Accuracy':>10} {'MacroF1':>10} {'WtF1':>10}")
    print(f"  {'-'*62}")
    for s in sorted(summaries, key=lambda s: -s["accuracy"]):
        print(
            f"  {s['name']:<22} {s['n']:>5}"
            f"  {s['accuracy']*100:>8.1f}%"
            f"  {s['macro_f1']*100:>8.1f}%"
            f"  {s['weighted_f1']*100:>8.1f}%"
        )

    out_path = os.path.join(EVAL_DIR, "schema_type_summary.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  ✅ Schema type summary saved → {os.path.relpath(out_path)}")

    return summaries


if __name__ == "__main__":
    run_full_schema_evaluation()
