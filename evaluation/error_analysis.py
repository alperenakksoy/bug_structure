import json
import os
from collections import defaultdict

RESULT_FILES = {
    "llama_8b_zero_shot":  "evaluation/results_llama_8b_zero_shot.json",
    "llama_8b_few_shot":   "evaluation/results_llama_8b_few_shot.json",
    "llama_70b_zero_shot": "evaluation/results_llama_70b_zero_shot.json",
    "llama_70b_few_shot":  "evaluation/results_llama_70b_few_shot.json",
    "qwen_zero_shot":   "evaluation/results_qwen_32b_zero_shot.json",
    "qwen_few_shot":    "evaluation/results_qwen_32b_few_shot.json",
}

CLASSES = ["low", "medium", "high", "critical"]


def load(path):
    with open(path) as f:
        return json.load(f)


def confusion_matrix(results):
    matrix = {t: {p: 0 for p in CLASSES + ["invalid"]} for t in CLASSES}
    for r in results:
        true = r.get("true_severity", "?")
        pred = r.get("severity", "?") if r.get("json_valid", False) else "invalid"
        if true in matrix:
            key = pred if pred in matrix[true] else "invalid"
            matrix[true][key] += 1
    return matrix


def print_confusion(name, matrix):
    cols = CLASSES + ["invalid"]
    print(f"\n  Confusion matrix — {name}")
    print(f"  {'True↓  Pred→':<14}" + "".join(f"{c:>10}" for c in cols))
    print(f"  {'-'*74}")
    for true in CLASSES:
        row = f"  {true:<14}"
        for pred in cols:
            row += f"{matrix[true].get(pred, 0):>10}"
        print(row)


def severity_bias(results):
    """Does the model systematically over/under-predict a severity?"""
    pred_counts  = defaultdict(int)
    true_counts  = defaultdict(int)
    for r in results:
        if r.get("json_valid"):
            pred_counts[r.get("severity", "?")] += 1
        true_counts[r.get("true_severity", "?")] += 1
    print(f"\n  Severity distribution (true vs predicted):")
    print(f"  {'Class':<12} {'True':>8} {'Predicted':>12} {'Δ':>8}")
    print(f"  {'-'*42}")
    for c in CLASSES:
        t = true_counts[c]
        p = pred_counts[c]
        print(f"  {c:<12} {t:>8} {p:>12} {p-t:>+8}")


def schema_type_distribution(results):
    counts = defaultdict(int)
    for r in results:
        counts[r.get("schema_type", "missing")] += 1
    print(f"\n  Schema type distribution:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<15} {v}")


def worst_errors(results, n=8):
    errors = [r for r in results if not r.get("correct_severity")]
    print(f"\n  Sample errors ({len(errors)} total):")
    for r in errors[:n]:
        true = r.get("true_severity", "?")
        pred = r.get("severity", "?")
        desc = r.get("description", "")[:75]
        hall = " [HALLUCINATED]" if r.get("hallucinated") else ""
        print(f"    true={true:<9} pred={pred:<9}{hall}")
        print(f"      \"{desc}\"")


def direction_analysis(results):
    """Are errors over-predictions or under-predictions?"""
    order = {c: i for i, c in enumerate(CLASSES)}
    over, under, same = 0, 0, 0
    for r in results:
        if r.get("correct_severity"):
            same += 1
            continue
        t = order.get(r.get("true_severity", ""), -1)
        p = order.get(r.get("severity", ""), -1)
        if t < 0 or p < 0:
            continue
        if p > t:
            over += 1
        else:
            under += 1
    total = over + under
    print(f"\n  Error direction (of {total} misclassifications):")
    print(f"    Over-predicted  (predicted higher severity): {over} ({over/total*100:.0f}%)" if total else "    No errors")
    print(f"    Under-predicted (predicted lower severity):  {under} ({under/total*100:.0f}%)" if total else "")


def run():
    all_stats = {}

    for name, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠  Missing: {path}")
            continue

        results = load(path)
        print(f"\n{'═'*65}")
        print(f"  {name}  (n={len(results)})")
        print(f"{'═'*65}")

        matrix = confusion_matrix(results)
        print_confusion(name, matrix)
        direction_analysis(results)
        severity_bias(results)
        schema_type_distribution(results)
        worst_errors(results)

        all_stats[name] = {
            "confusion_matrix": matrix,
            "n": len(results),
        }

    # Save
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/error_analysis.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print("\n\n✅  Error analysis saved → evaluation/error_analysis.json")


if __name__ == "__main__":
    run()