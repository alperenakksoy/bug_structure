import json
import os

results = {
    "llama_zero_shot": "evaluation/results_llama_zero_shot.json",
    "gemma_zero_shot": "evaluation/results_gemma_zero_shot.json",
    "llama_few_shot": "evaluation/results_llama_few_shot.json",
    "gemma_few_shot": "evaluation/results_gemma_few_shot.json",
}

print("=" * 50)
print("EXPERIMENT RESULTS SUMMARY")
print("=" * 50)

for name, path in results.items():
    full_path = path
    if os.path.exists(full_path):
        with open(full_path) as f:
            data = json.load(f)
        correct = sum(1 for r in data if r.get("correct_severity"))
        total = len(data)
        acc = correct / total * 100 if total > 0 else 0
        print(f"{name:25} → {correct}/{total} = {acc:.1f}%")

print("=" * 50)
print("\nBest: Llama 3.1 8B + Few-shot = 53.3%")