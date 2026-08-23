"""Measures the zero-shot base-model accuracy on GSM8K -- the REI reference
point (`--baseline_accuracy` in scripts/run_experiment.py and
scripts/run_matrix.sh).

Inference-only: no training, no gradients, no optimizer state. Strictly
lighter than every training run this pipeline has already done on a free
T4 (which all fit fine), so this runs there too -- no need to spend
rented-GPU budget on it.

Checkpointed and resumable: each example's result is appended to
--records_path (JSONL, one line per example) as soon as it's generated,
not accumulated in memory and written once at the end. A disconnect at
example 41, 400, or 1300 loses nothing before that point -- rerunning the
exact same command picks up where it left off, skipping indices already
present in the records file. Point --records_path at a Drive-mounted path
(e.g. via a `results/` symlink into Drive) to survive a Colab VM disconnect
entirely, not just a script crash.

Usage:
    # Quick sanity check first (a few minutes):
    python scripts/measure_baseline.py --n_eval_samples 50

    # Full run once the quick check looks sane (can take a couple hours on
    # a free T4 -- keep the Colab tab active so the session doesn't idle
    # out; if it does, just rerun the same command to resume):
    python scripts/measure_baseline.py

    # Start over from scratch under new settings instead of resuming
    # (wipes --records_path first):
    python scripts/measure_baseline.py --restart
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import load_gsm8k
from src.evaluate import answers_match, extract_final_answer
from src.models import load_model, load_tokenizer

MODEL_ID = "google/gemma-4-E2B-it"
RECORDS_PATH_DEFAULT = "results/baseline_records.jsonl"
SUMMARY_PATH = "results/baseline_accuracy.json"


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens):
    # add_special_tokens=False: prompt already went through the tokenizer's
    # chat template, which embeds BOS/turn tokens itself.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return {
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "n_generated_tokens": int(generated_ids.shape[0]),
        "hit_token_limit": bool(generated_ids.shape[0] >= max_new_tokens),
    }


def load_existing_records(records_path):
    """Returns {index: record} for whatever's already on disk, or {} if the
    file doesn't exist yet. index is the example's 1-based position in the
    deterministic GSM8K test-split ordering, so it stays valid across runs
    with different --n_eval_samples (a smaller run's records are a prefix
    of a larger run's)."""
    if not os.path.exists(records_path):
        return {}
    records = {}
    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["index"]] = record
    return records


def check_config_matches(records_path, existing_records, model_id, max_new_tokens):
    if not existing_records:
        return
    sample = next(iter(existing_records.values()))
    if sample.get("model_id") != model_id or sample.get("max_new_tokens") != max_new_tokens:
        raise ValueError(
            f"{records_path} has records generated with model_id={sample.get('model_id')!r}, "
            f"max_new_tokens={sample.get('max_new_tokens')} -- this run is "
            f"model_id={model_id!r}, max_new_tokens={max_new_tokens}. Resuming would silently "
            f"mix incompatible results. Pass --restart to wipe the records file and start over, "
            f"or --records_path to write somewhere new."
        )


def append_record(records_path, record):
    with open(records_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def warn_if_not_symlinked(records_path):
    """`ln -s SOURCE DEST` silently creates the link *inside* DEST instead
    of replacing it if DEST already exists as a real directory -- which
    `results/` does here, since it's tracked in this repo. That means a
    Drive-mount symlink setup can "succeed" (no error) while every write
    still goes to /content/'s ephemeral local disk, invisible until a Colab
    disconnect wipes it. Warn loudly up front rather than discover this
    after losing hours of generation."""
    parent = os.path.dirname(records_path) or "."
    if os.path.isdir(parent) and not os.path.islink(parent):
        print(
            f"WARNING: {parent}/ is a plain local directory, not a symlink. On Colab, "
            f"/content/ is wiped on every disconnect -- if you intended {parent}/ to "
            f"point at Drive (or other persistent storage), your symlink setup didn't "
            f"take (commonly because {parent}/ already existed, so `ln -s` linked "
            f"*inside* it instead of replacing it). Verify with "
            f"`ls -la {parent}` (look for '->') before a long run, or everything "
            f"written this session is lost on the next disconnect."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval_samples", type=int, default=None, help="Subset size; omit for the full 1,319-example test set")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-example progress output")
    parser.add_argument("--records_path", default=RECORDS_PATH_DEFAULT, help="Per-example JSONL checkpoint file")
    parser.add_argument("--restart", action="store_true", help="Wipe --records_path and start over instead of resuming")
    args = parser.parse_args()

    warn_if_not_symlinked(args.records_path)

    if os.path.dirname(args.records_path):
        os.makedirs(os.path.dirname(args.records_path), exist_ok=True)

    if args.restart and os.path.exists(args.records_path):
        os.remove(args.records_path)
        print(f"--restart: removed existing {args.records_path}")

    existing_records = load_existing_records(args.records_path)
    check_config_matches(args.records_path, existing_records, MODEL_ID, args.max_new_tokens)
    if existing_records:
        print(f"Found {len(existing_records)} existing record(s) in {args.records_path} -- resuming.")

    print(f"Loading base model ({MODEL_ID}, bf16, no LoRA, no quantization -- unmodified weights)...")
    config = {"model_id": MODEL_ID, "bf16": True}
    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)
    model.eval()

    eval_ds = load_gsm8k(tokenizer, "test", n_samples=args.n_eval_samples)
    n = len(eval_ds)
    print(f"Evaluating zero-shot on {n} GSM8K test examples (indices 1..{n})...")

    n_skipped = 0
    for i in range(1, n + 1):
        if i in existing_records:
            n_skipped += 1
            continue

        example = eval_ds[i - 1]
        reference = extract_final_answer(example["answer"])
        result = generate(model, tokenizer, example["prompt"], max_new_tokens=args.max_new_tokens)
        predicted = extract_final_answer(result["text"])
        correct = answers_match(predicted, reference)

        record = {
            "index": i,
            "question": example["question"],
            "reference": reference,
            "predicted": predicted,
            "correct": correct,
            "generated_text": result["text"],
            "n_generated_tokens": result["n_generated_tokens"],
            "hit_token_limit": result["hit_token_limit"],
            "model_id": MODEL_ID,
            "max_new_tokens": args.max_new_tokens,
        }
        append_record(args.records_path, record)
        existing_records[i] = record

        if not args.quiet:
            flag = " [TRUNCATED]" if result["hit_token_limit"] else ""
            print(f"[{i}/{n}] correct={correct} pred={predicted} ref={reference}{flag}")

    if n_skipped:
        print(f"\nSkipped {n_skipped} example(s) already present in {args.records_path}.")

    all_records = [existing_records[i] for i in range(1, n + 1)]
    correct_count = sum(r["correct"] for r in all_records)
    accuracy = correct_count / n if n else 0.0
    n_truncated_total = sum(r["hit_token_limit"] for r in all_records)

    print(f"\nZero-shot baseline accuracy: {accuracy:.4f} ({correct_count}/{n})")
    if n_truncated_total:
        print(
            f"WARNING: {n_truncated_total}/{n} examples hit max_new_tokens={args.max_new_tokens} "
            f"before stopping naturally -- their predictions may be unreliable. See "
            f"{args.records_path} for per-example hit_token_limit flags, and "
            f"scripts/inspect_by_index.py to look closer."
        )

    with open(SUMMARY_PATH, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "n_examples": n,
                "accuracy": accuracy,
                "max_new_tokens": args.max_new_tokens,
                "n_truncated": n_truncated_total,
                "records_path": args.records_path,
            },
            f,
            indent=2,
        )
    print(f"Saved summary to {SUMMARY_PATH}, per-example records in {args.records_path}")
    print("\nUse this as --baseline_accuracy in scripts/run_experiment.py / GG_BASELINE_ACCURACY for scripts/run_matrix.sh:")
    print(f"  --baseline_accuracy {accuracy}")


if __name__ == "__main__":
    main()
