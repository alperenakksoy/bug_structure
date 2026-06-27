import json
import os
from collections import defaultdict
from typing import List, Dict

RESULT_FILES = {
    "llama_8b_zero":   "evaluation/results_llama_8b_zero_shot.json",
    "llama_8b_few":    "evaluation/results_llama_8b_few_shot.json",
    "llama_70b_zero":  "evaluation/results_llama_70b_zero_shot.json",
    "llama_70b_few":   "evaluation/results_llama_70b_few_shot.json",
    "qwen_zero_shot": "evaluation/results_qwen_32b_zero_shot.json",
    "qwen_few_shot": "evaluation/results_qwen_32b_few_shot.json",
}

CLASSES = ["low", "medium", "high", "critical"]


def precision_recall_f1(results: List[Dict], label: str):
    tp = sum(1 for r in results if r.get("severity") == label and r.get("true_severity") == label)
    fp = sum(1 for r in results if r.get("severity") == label and r.get("true_severity") != label)
    fn = sum(1 for r in results if r.get("severity") != label and r.get("true_severity") == label)

    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    support = tp + fn
    return p, r, f1, support


def macro_f1(results):
    f1s = [precision_recall_f1(results, c)[2] for c in CLASSES]
    return sum(f1s) / len(f1s)


def weighted_f1(results):
    total = len(results)
    if total == 0:
        return 0.0
    score = 0.0
    for c in CLASSES:
        _, _, f1, support = precision_recall_f1(results, c)
        score += f1 * support
    return score / total


def json_validity_rate(results):
    valid = sum(1 for r in results if r.get("json_valid", False))
    return valid / len(results) if results else 0.0


def hallucination_rate(results):
    hallucinated = sum(1 for r in results if r.get("hallucinated", False))
    return hallucinated / len(results) if results else 0.0


def accuracy(results):
    correct = sum(1 for r in results if r.get("correct_severity", False))
    return correct / len(results) if results else 0.0


# ── Confusion matrix ─────────────────────────────────────────────────────────
def confusion_matrix(results) -> Dict[str, Dict[str, int]]:
    matrix = {true: {pred: 0 for pred in CLASSES + ["other"]} for true in CLASSES}
    for r in results:
        true = r.get("true_severity", "?")
        pred = r.get("severity", "?")
        if true in matrix:
            if pred in matrix[true]:
                matrix[true][pred] += 1
            else:
                matrix[true]["other"] += 1
    return matrix


def print_confusion_matrix(matrix):
    header = f"{'True \\ Pred':<12}" + "".join(f"{c:>10}" for c in CLASSES)
    print(header)
    print("-" * (12 + 10 * len(CLASSES)))
    for true_class in CLASSES:
        row = f"{true_class:<12}"
        for pred_class in CLASSES:
            row += f"{matrix[true_class].get(pred_class, 0):>10}"
        print(row)


# ── Error analysis ───────────────────────────────────────────────────────────

def error_analysis(results, top_n: int = 5):
    errors = [r for r in results if not r.get("correct_severity", True)]

    # Most common true→predicted mistakes
    mistake_counts = defaultdict(int)
    for r in errors:
        key = f"{r.get('true_severity')} → {r.get('severity', '?')}"
        mistake_counts[key] += 1

    sorted_mistakes = sorted(mistake_counts.items(), key=lambda x: -x[1])

    print(f"\n  Most common misclassifications (out of {len(errors)} errors):")
    for pattern, count in sorted_mistakes[:top_n]:
        print(f"    {pattern:<25} {count} times")

    # Examples of worst mistakes
    print(f"\n  Sample error cases:")
    for r in errors[:top_n]:
        true = r.get("true_severity", "?")
        pred = r.get("severity", "?")
        desc = r.get("description", "")[:70]
        print(f"    [true={true} | pred={pred}] {desc}")


# ── Full report ───────────────────────────────────────────────────────────────

def evaluate_experiment(name: str, results: List[Dict]):
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    print(f"  Samples       : {len(results)}")
    print(f"  Accuracy      : {accuracy(results)*100:.1f}%")
    print(f"  Macro F1      : {macro_f1(results)*100:.1f}%")
    print(f"  Weighted F1   : {weighted_f1(results)*100:.1f}%")
    print(f"  JSON Valid    : {json_validity_rate(results)*100:.1f}%")
    print(f"  Hallucination : {hallucination_rate(results)*100:.1f}%")

    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    for c in CLASSES:
        p, r, f1, support = precision_recall_f1(results, c)
        print(f"  {c:<12} {p*100:>9.1f}% {r*100:>9.1f}% {f1*100:>9.1f}% {support:>10}")

    print(f"\n  Confusion matrix:")
    matrix = confusion_matrix(results)
    print_confusion_matrix(matrix)

    error_analysis(results)

    return {
        "name":             name,
        "n":                len(results),
        "accuracy":         round(accuracy(results), 4),
        "macro_f1":         round(macro_f1(results), 4),
        "weighted_f1":      round(weighted_f1(results), 4),
        "json_valid_rate":  round(json_validity_rate(results), 4),
        "hallucination_rate": round(hallucination_rate(results), 4),
        "per_class": {
            c: {
                "precision": round(precision_recall_f1(results, c)[0], 4),
                "recall":    round(precision_recall_f1(results, c)[1], 4),
                "f1":        round(precision_recall_f1(results, c)[2], 4),
                "support":   precision_recall_f1(results, c)[3],
            }
            for c in CLASSES
        }
    }


def run_full_evaluation():
    all_summaries = []

    for name, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠ Missing: {path} — skipping")
            continue
        with open(path) as f:
            results = json.load(f)
        summary = evaluate_experiment(name, results)
        all_summaries.append(summary)

    # ── Comparative summary table ─────────────────────────────────────────────
    print(f"\n\n{'═'*75}")
    print("  COMPARATIVE SUMMARY")
    print(f"{'═'*75}")
    print(f"  {'Experiment':<22} {'Acc':>7} {'MacroF1':>9} {'WtF1':>7} {'JSON%':>7} {'Hall%':>7}")
    print(f"  {'-'*65}")
    for s in all_summaries:
        print(
            f"  {s['name']:<22}"
            f"  {s['accuracy']*100:>5.1f}%"
            f"  {s['macro_f1']*100:>7.1f}%"
            f"  {s['weighted_f1']*100:>5.1f}%"
            f"  {s['json_valid_rate']*100:>5.1f}%"
            f"  {s['hallucination_rate']*100:>5.1f}%"
        )

    # Save machine-readable summary
    out_path = "evaluation/full_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  ✅ Full summary saved → {out_path}")

    return all_summaries


if __name__ == "__main__":
    run_full_evaluation()