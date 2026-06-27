import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.extractor_v2 import run_experiment
from evaluation.evaluate import run_full_evaluation
from evaluation.error_analysis import run as run_error_analysis

EXPERIMENTS = [
    ("llama_8b",  "zero_shot"),
    ("llama_8b",  "few_shot"),
    ("llama_70b", "zero_shot"),
    ("llama_70b", "few_shot"),
    ("qwen",   "zero_shot"),
    ("qwen",   "few_shot"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--n", type=int, default=150)
    args = parser.parse_args()

    if not args.eval_only:
        print(f"\n🚀  Running {len(EXPERIMENTS)} experiments × {args.n} samples each")
        print(f"    Estimated time: ~{len(EXPERIMENTS) * args.n * 0.5 / 60:.0f} min (0.5s/call)\n")
        for model_key, prompt_type in EXPERIMENTS:
            run_experiment(n_samples=args.n, model_key=model_key, prompt_type=prompt_type)

    print("\n\n📊  Running evaluation...")
    run_full_evaluation()

    print("\n\n🔍  Running error analysis...")
    run_error_analysis()

    print("\n\n✅  July 1 milestone complete.")
    print("    Results in: evaluation/")


if __name__ == "__main__":
    main()