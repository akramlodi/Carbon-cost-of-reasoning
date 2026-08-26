"""Measures the zero-shot base-model accuracy on GSM8K -- the REI reference
point (`--baseline_accuracy` in scripts/run_experiment.py and
scripts/run_matrix.sh).

Inference-only: no training, no gradients, no optimizer state. Generation
is batched (left-padded, decoder-only-model-correct) rather than one
example at a time -- at ~1 min/example unbatched, a full 1,319-example
GSM8K run takes ~22 hours, impractical on a billed instance. Designed for
AWS g5.xlarge (1x A10G, 24GB) running google/gemma-4-E2B-it in bf16
(~10.2GB model): the constraint there is speed, not memory, so this
doesn't reach for quantization -- just batching.

Checkpointed and resumable: each completed example's result is appended to
--records_path (JSONL, one line per example) as its batch finishes, not
accumulated in memory and written once at the end. A crash or disconnect
loses only the in-flight batch -- rerunning the exact same command skips
every index already present and resumes from there. This matters doubly
now: on a billed instance, a lost run costs money, not just time.

Usage:
    # Quick sanity check first (a few minutes):
    python scripts/measure_baseline.py --n_eval_samples 50 --batch_size 16

    # Full run once the quick check looks sane:
    python scripts/measure_baseline.py --batch_size 16

    # Start over from scratch under new settings instead of resuming
    # (wipes --records_path first):
    python scripts/measure_baseline.py --restart
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import load_gsm8k
from src.evaluate import answers_match, extract_final_answer
from src.models import load_model, load_tokenizer

MODEL_ID = "google/gemma-4-E2B-it"
RECORDS_PATH_DEFAULT = "results/baseline_records.jsonl"
SUMMARY_PATH = "results/baseline_accuracy.json"


def normalize_eos_ids(model, tokenizer):
    """model.generation_config.eos_token_id can be a single int or a list
    (Gemma chat models typically stop on more than one valid token, e.g.
    both <eos> and <end_of_turn>) -- normalize to a set for membership
    checks. Falls back to the tokenizer's eos_token_id if the generation
    config doesn't define one."""
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    return set(eos)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, eos_token_ids):
    """Batched generation with left-padding.

    Left-padding means every row's real prompt content ends at the same
    absolute position (input_ids.shape[1]) regardless of how much padding
    precedes it -- that's the whole mechanism that makes batched
    decoder-only generation possible, and it's why a single shared slice
    index (input_length) correctly isolates the generated continuation for
    every row; right-padding would make that slice point vary per row and
    silently produce garbage.

    What genuinely does vary per row, and can't be read from a single
    shared value, is *where each row's own generation actually stopped*.
    generate() runs the whole batch in lockstep and pads rows that finish
    early with pad_token_id until every row is done (or max_new_tokens is
    hit) -- so a row that stopped early still has a slice of length
    max_new_tokens in the output tensor, just with trailing padding after
    its real content. We recover the real per-row length and truncation
    status by scanning each row's generated slice for the first occurrence
    of any valid eos token: found -> stopped naturally; not found -> that
    row was still generating when the batch-wide budget ran out.
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    results = []
    for row in output_ids:
        gen_ids = row[input_length:].tolist()
        eos_pos = next((j for j, t in enumerate(gen_ids) if t in eos_token_ids), None)
        if eos_pos is not None:
            n_generated_tokens = eos_pos + 1
            hit_token_limit = False
        else:
            n_generated_tokens = len(gen_ids)
            hit_token_limit = True
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({
            "text": text,
            "n_generated_tokens": n_generated_tokens,
            "hit_token_limit": hit_token_limit,
        })
    return results


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
    """batch_size is deliberately NOT checked -- it's a pure throughput
    knob (e.g. backing off after an OOM mid-run) and doesn't change what's
    being asked of the model or how it's scored, unlike model_id or
    max_new_tokens."""
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
    persistent-storage symlink setup (Drive on Colab, or any mount) can
    "succeed" with no error while every write still goes to local/ephemeral
    disk. Warn loudly up front rather than discover this after losing a
    run -- doubly so on a billed instance."""
    parent = os.path.dirname(records_path) or "."
    if os.path.isdir(parent) and not os.path.islink(parent):
        print(
            f"WARNING: {parent}/ is a plain local directory, not a symlink. If you intended "
            f"{parent}/ to point at persistent/mounted storage, your symlink setup didn't take "
            f"(commonly because {parent}/ already existed, so `ln -s` linked *inside* it instead "
            f"of replacing it). Verify with `ls -la {parent}` (look for '->') before a long run."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval_samples", type=int, default=None, help="Subset size; omit for the full 1,319-example test set")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16, help="Back off to 8 or 4 if this OOMs")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-example progress output")
    parser.add_argument("--records_path", default=RECORDS_PATH_DEFAULT, help="Per-example JSONL checkpoint file")
    parser.add_argument("--restart", action="store_true", help="Wipe --records_path and start over instead of resuming")
    parser.add_argument("--log_every_n_batches", type=int, default=1, help="Progress-line frequency, for tmux-attach visibility on long runs")
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
    tokenizer.padding_side = "left"  # required for correct batched generation with a decoder-only model
    if tokenizer.pad_token is None:  # load_tokenizer already guarantees this; redundant but cheap
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(config)
    model.eval()

    eos_token_ids = normalize_eos_ids(model, tokenizer)

    eval_ds = load_gsm8k(tokenizer, "test", n_samples=args.n_eval_samples)
    n = len(eval_ds)
    pending_indices = [i for i in range(1, n + 1) if i not in existing_records]
    print(f"{n} GSM8K test examples total, {len(pending_indices)} pending, batch_size={args.batch_size}.")

    start_time = time.time()
    n_done_this_run = 0

    for batch_start in range(0, len(pending_indices), args.batch_size):
        batch_indices = pending_indices[batch_start:batch_start + args.batch_size]
        examples = [eval_ds[i - 1] for i in batch_indices]
        prompts = [example["prompt"] for example in examples]

        batch_results = generate_batch(model, tokenizer, prompts, args.max_new_tokens, eos_token_ids)

        for idx, example, gen in zip(batch_indices, examples, batch_results):
            reference = extract_final_answer(example["answer"])
            predicted = extract_final_answer(gen["text"])
            correct = answers_match(predicted, reference)

            record = {
                "index": idx,
                "question": example["question"],
                "reference": reference,
                "predicted": predicted,
                "correct": correct,
                "generated_text": gen["text"],
                "n_generated_tokens": gen["n_generated_tokens"],
                "hit_token_limit": gen["hit_token_limit"],
                "model_id": MODEL_ID,
                "max_new_tokens": args.max_new_tokens,
            }
            append_record(args.records_path, record)
            existing_records[idx] = record

            if not args.quiet:
                flag = " [TRUNCATED]" if gen["hit_token_limit"] else ""
                print(f"[{idx}/{n}] correct={correct} pred={predicted} ref={reference}{flag}")

        n_done_this_run += len(batch_indices)
        batch_num = batch_start // args.batch_size + 1
        is_last_batch = batch_start + args.batch_size >= len(pending_indices)
        if batch_num % args.log_every_n_batches == 0 or is_last_batch:
            elapsed = time.time() - start_time
            rate = n_done_this_run / elapsed if elapsed > 0 else 0.0
            remaining = len(pending_indices) - n_done_this_run
            eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
            print(
                f"-- progress: {n_done_this_run}/{len(pending_indices)} new this run "
                f"({len(existing_records)}/{n} total done), {rate:.2f} ex/sec, "
                f"ETA {eta_min:.1f} min --"
            )

    all_records = [existing_records[i] for i in range(1, n + 1)]
    correct_count = sum(r["correct"] for r in all_records)
    accuracy = correct_count / n if n else 0.0
    n_truncated = sum(r["hit_token_limit"] for r in all_records)
    truncation_rate = n_truncated / n if n else 0.0

    print(f"\nZero-shot baseline accuracy: {accuracy:.4f} ({correct_count}/{n})")
    print(f"Truncation rate: {truncation_rate:.4f} ({n_truncated}/{n} hit max_new_tokens={args.max_new_tokens})")
    if n_truncated:
        print("This truncation rate belongs in the paper's limitations section regardless of its value.")

    with open(SUMMARY_PATH, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "n_examples": n,
                "accuracy": accuracy,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "n_truncated": n_truncated,
                "truncation_rate": truncation_rate,
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
