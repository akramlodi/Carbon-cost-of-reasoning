"""Measures the zero-shot base-model accuracy on GSM8K -- the REI reference
point (`--baseline_accuracy` in scripts/run_experiment.py and
scripts/run_matrix.sh).

Inference-only: no training, no gradients, no optimizer state. Strictly
lighter than every training run this pipeline has already done on a free
T4 (which all fit fine), so this runs there too -- no need to spend
rented-GPU budget on it. It just needs to complete, not be fast; expect it
to take a while on the full test set (1,319 examples, one generation call
each) since generation isn't batched here (see src/evaluate.py).

Usage:
    # Quick sanity check first (a few minutes):
    python scripts/measure_baseline.py --n_eval_samples 50

    # Full run once the quick check looks sane (can take a couple hours on
    # a free T4 -- keep the Colab tab active so the session doesn't idle out):
    python scripts/measure_baseline.py
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_gsm8k
from src.evaluate import evaluate_model
from src.models import load_model, load_tokenizer

MODEL_ID = "google/gemma-4-E2B-it"
OUT_PATH = "results/baseline_accuracy.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval_samples", type=int, default=None, help="Subset size; omit for the full 1,319-example test set")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-example progress output")
    args = parser.parse_args()

    print(f"Loading base model ({MODEL_ID}, bf16, no LoRA, no quantization -- unmodified weights)...")
    config = {"model_id": MODEL_ID, "bf16": True}
    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)

    eval_ds = load_gsm8k(tokenizer, "test", n_samples=args.n_eval_samples)
    print(f"Evaluating zero-shot on {len(eval_ds)} GSM8K test examples...")

    accuracy, records = evaluate_model(
        model, tokenizer, eval_ds, max_new_tokens=args.max_new_tokens, verbose=not args.quiet
    )

    print(f"\nZero-shot baseline accuracy: {accuracy:.4f} ({sum(r['correct'] for r in records)}/{len(records)})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "n_examples": len(records),
                "accuracy": accuracy,
                "max_new_tokens": args.max_new_tokens,
            },
            f,
            indent=2,
        )
    print(f"Saved to {OUT_PATH}")
    print(f"\nUse this as --baseline_accuracy in scripts/run_experiment.py / GG_BASELINE_ACCURACY for scripts/run_matrix.sh:")
    print(f"  --baseline_accuracy {accuracy}")


if __name__ == "__main__":
    main()
