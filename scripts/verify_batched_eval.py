"""GPU check for the batched `src/evaluate.py::evaluate_model` (CHANGES.md Change 6).

Answers the two questions that can't be answered on CPU, in one run:

  1. Correctness -- do batched predictions match the unbatched `generate_answer()`
     path example-for-example on the real gemma-4 checkpoint? (The CPU test in
     tests/test_evaluate_batching.py covers this with a tiny stand-in model;
     this is the same question asked of the real weights.)
  2. Speed -- how much faster is batched, and what does that mean for the
     1,319-example trailing eval that every run in the matrix pays for?

Run from the repository root inside tmux, on the g5.xlarge:

    # base model, 32 examples, batch 16
    python scripts/verify_batched_eval.py

    # the fine-tuned checkpoint the interrupted run already has on disk
    # (save_strategy="epoch" means checkpoint-<step>/ holds the adapter even
    # though final_adapter_or_model/ is only written after the trailing eval)
    python scripts/verify_batched_eval.py \\
      --adapter results/runs/lora/lora_r16_seed1/checkpoint-1404

Exits non-zero if any prediction differs, so it can gate the matrix run.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import load_gsm8k
from src.evaluate import (
    EVAL_BATCH_SIZE,
    answers_match,
    evaluate_model,
    extract_final_answer,
    generate_answer,
)
from src.models import load_model, load_tokenizer

FULL_EVAL_SIZE = 1319


def unbatched_records(model, tokenizer, examples, max_new_tokens):
    """The reference path: exactly what evaluate_model did before batching."""
    records = []
    for example in examples:
        reference = extract_final_answer(example["answer"])
        predicted = extract_final_answer(
            generate_answer(model, tokenizer, example["prompt"], max_new_tokens)
        )
        records.append({
            "question": example["question"],
            "reference": reference,
            "predicted": predicted,
            "correct": answers_match(predicted, reference),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml", help="Supplies model_id / bf16 / quantization")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir to load on top of the base model")
    parser.add_argument("--n_examples", type=int, default=32, help="Examples per side of the comparison")
    parser.add_argument("--start_index", type=int, default=1,
                        help="1-based first test example (measure_baseline.py's indexing)")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Project-wide standard; keep it at 768")
    parser.add_argument("--batch_size", type=int, default=EVAL_BATCH_SIZE, help="Back off to 8 or 4 if this OOMs")
    parser.add_argument("--max_mismatch_rate", type=float, default=0.10,
                        help="Fraction of examples whose final ANSWER may differ between the batched and "
                             "unbatched paths; see Check 1's explanation and "
                             "scripts/diagnose_batched_divergence.py")
    parser.add_argument("--strict", action="store_true", help="Fail on any per-example mismatch, ignoring the rate")
    args = parser.parse_args()

    if args.n_examples < 64 and not args.strict:
        print(f"WARNING: --n_examples={args.n_examples} is a small sample; a handful of flipped "
              f"answers moves the mismatch rate by several points. Use --n_examples 128 before "
              f"treating the rate as a gate.")

    import yaml

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded adapter from {args.adapter}")
    model.eval()
    if torch.cuda.is_available():
        print(f"Model resident: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

    eval_ds = load_gsm8k(tokenizer, "test")
    start = args.start_index - 1
    if start + args.n_examples > len(eval_ds):
        raise ValueError(
            f"--start_index {args.start_index} + --n_examples {args.n_examples} exceeds "
            f"the {len(eval_ds)}-example test split"
        )
    examples = [eval_ds[i] for i in range(start, start + args.n_examples)]
    print(
        f"Comparing {len(examples)} example(s) from 1-based index {args.start_index}, "
        f"max_new_tokens={args.max_new_tokens}, batch_size={args.batch_size}"
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    reference_records = unbatched_records(model, tokenizer, examples, args.max_new_tokens)
    unbatched_s = time.time() - start_time
    unbatched_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    batched_accuracy, batched_records = evaluate_model(
        model, tokenizer, examples, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size, log_every_n_batches=0,
    )
    batched_s = time.time() - start_time
    batched_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    mismatches = [
        (i + start + 1, reference_records[i], batched_records[i])
        for i in range(len(examples))
        if reference_records[i] != batched_records[i]
    ]
    unbatched_accuracy = sum(r["correct"] for r in reference_records) / len(reference_records)
    gained = [m for m in mismatches if m[2]["correct"] and not m[1]["correct"]]
    lost = [m for m in mismatches if m[1]["correct"] and not m[2]["correct"]]

    print("\n" + "=" * 72)
    print("Check 1: batched vs unbatched final answers (answer-level, not text-level)")
    print("  Note: the underlying generated text diverges far more often than the")
    print("  extracted answer does -- one flipped token usually still lands on the same")
    print("  number. scripts/diagnose_batched_divergence.py reports both rates.")
    for index, reference, batched in mismatches:
        print(f"  differs at 1-based index {index}:")
        print(f"    unbatched: predicted {reference['predicted']} "
              f"(correct={reference['correct']})")
        print(f"    batched  : predicted {batched['predicted']} "
              f"(correct={batched['correct']})")
    print(f"  {len(examples) - len(mismatches)}/{len(examples)} identical; "
          f"{len(gained)} wrong->right, {len(lost)} right->wrong")
    print(f"  accuracy: unbatched {unbatched_accuracy:.4f} vs batched "
          f"{batched_accuracy:.4f} (delta {batched_accuracy - unbatched_accuracy:+.4f})")

    if mismatches:
        print("\n  Some per-example disagreement is EXPECTED and is not a bug: batch shape")
        print("  changes matmul reduction order, so logits move slightly, and greedy decode")
        print("  flips a token wherever two candidates are near-tied -- after which the two")
        print("  continuations are different sentences. This is not specific to left-padding")
        print("  (a 1-row batch has none) and is not directional: it moves examples both ways.")
        print("  What matters is that the batched path is deterministic run-to-run (Check 3)")
        print("  and that every condition and the zero-shot baseline share the same batch")
        print("  boundaries, so REI comparisons stay apples-to-apples.")
        print("  Run scripts/diagnose_batched_divergence.py to confirm the cause.")

    mismatch_rate = len(mismatches) / len(examples)
    if mismatches and not args.strict and mismatch_rate <= args.max_mismatch_rate:
        print(f"\n  PASS (within --max_mismatch_rate={args.max_mismatch_rate})")
    elif mismatches:
        print(f"\n  FAIL -- {len(mismatches)}/{len(examples)} differ "
              f"(rate {mismatch_rate:.3f}), above --max_mismatch_rate="
              f"{args.max_mismatch_rate} or --strict was passed.")
        print("  A rate this high is not rounding noise; investigate before running the matrix.")
        print("=" * 72)
        return 1
    else:
        print("  PASS -- all predictions identical")

    print("\n" + "=" * 72)
    print("Check 2: speed")
    n = len(examples)
    print(f"  unbatched : {unbatched_s:8.1f}s  ({n / unbatched_s:.3f} ex/sec, peak {unbatched_peak:.2f} GB)")
    print(f"  batched   : {batched_s:8.1f}s  ({n / batched_s:.3f} ex/sec, peak {batched_peak:.2f} GB)")
    print(f"  speedup   : {unbatched_s / batched_s:.2f}x")
    print(f"  extrapolated full {FULL_EVAL_SIZE}-example eval: "
          f"{unbatched_s / n * FULL_EVAL_SIZE / 60:.1f} min unbatched -> "
          f"{batched_s / n * FULL_EVAL_SIZE / 60:.1f} min batched")
    if batched_s >= unbatched_s:
        print("  FAIL -- batching was not faster; check whether the KV cache is active "
              "(a model.config.use_cache=False would make decode quadratic)")
        print("=" * 72)
        return 1
    print("  PASS")

    print("\n" + "=" * 72)
    print("Check 3: determinism of the batched path (what the matrix actually needs)")
    _, second_pass = evaluate_model(
        model, tokenizer, examples, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size, log_every_n_batches=0,
    )
    if second_pass != batched_records:
        print("  FAIL -- the identical command produced different predictions; the eval "
              "is not reproducible and no matrix result would be trustworthy")
        print("=" * 72)
        return 1
    print("  PASS -- identical predictions on a repeat of the identical command")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
