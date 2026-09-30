"""Diagnose why batched and unbatched greedy generation disagree on a few examples.

`scripts/verify_batched_eval.py` compares batched against unbatched predictions
and can report mismatches (CHANGES.md Change 6). This script answers *why*,
with the experiments that separate the two candidate causes:

  A. A bug in the batching -- wrong slice, leaked padding, mispaired rows.
  B. Float-level nondeterminism across batch shapes. Different batch sizes
     change matmul shapes, which changes reduction order, which perturbs
     logits slightly. Greedy decode is discontinuous, so a near-tie flips one
     token and the two continuations diverge from there.

Three checks, each of which would have looked different under (A) and (B):

  C. Determinism. The identical command twice must give identical output.
     This is what the experiment actually needs: the batched path has to be
     reproducible, not byte-identical to the unbatched one.
  D. 1-row batch (no left-padding at all) vs the full batch. If they also
     disagree, the disagreement survives with zero padding, so it cannot be
     about how padding is being sliced.
  E. Same batch size, different neighbours. Neither the prompt nor its
     padding changed -- only which other rows shared the batch. If the same
     prompt's output changes anyway, that is arithmetic order (B).

Batching is what the zero-shot baseline (0.8877) was already measured with,
so under (B) every condition and the baseline share identical batch
boundaries and remain comparable; the cost is a handful of per-example flips
on both sides of a comparison, not a directional bias.

Run from the repository root on the g5.xlarge, inside tmux:
    python scripts/diagnose_batched_divergence.py \\
      --adapter results/runs/lora/lora_r16_seed1/checkpoint-1404
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from src.data import load_gsm8k
from src.evaluate import (
    EVAL_BATCH_SIZE,
    extract_final_answer,
    generate_batch,
    normalize_eos_ids,
)
from src.models import load_model, load_tokenizer


def truncate_at_eos(token_ids, eos_token_ids):
    for i, token in enumerate(token_ids):
        if token in eos_token_ids:
            return token_ids[: i + 1]
    return token_ids


def generate_token_ids(model, tokenizer, prompts, max_new_tokens, eos_token_ids):
    """Left-padded batched generation, but returning raw sequences and scores
    so the divergence can be located at token resolution."""
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        tokenizer.padding_side = previous_padding_side
    sequences = [
        truncate_at_eos(row[input_length:].tolist(), eos_token_ids)
        for row in out.sequences
    ]
    return sequences, out.scores


def first_divergence(a_ids, b_ids):
    for i, (a, b) in enumerate(zip(a_ids, b_ids)):
        if a != b:
            return i
    return min(len(a_ids), len(b_ids)) if len(a_ids) != len(b_ids) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--n_examples", type=int, default=32)
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--max_report", type=int, default=5, help="Mismatching examples to detail")
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded adapter from {args.adapter}")
    model.eval()

    eos_token_ids = normalize_eos_ids(model, tokenizer)
    eval_ds = load_gsm8k(tokenizer, "test")
    start = args.start_index - 1
    if start + args.n_examples > len(eval_ds):
        raise ValueError("--start_index + --n_examples exceeds the test split")
    examples = [eval_ds[i] for i in range(start, start + args.n_examples)]
    prompts = [ex["prompt"] for ex in examples]
    print(f"{len(examples)} example(s) from 1-based index {args.start_index}\n")

    def batched(ps, batch_size):
        out = []
        for i in range(0, len(ps), batch_size):
            out.extend(generate_batch(model, tokenizer, ps[i:i + batch_size],
                                      args.max_new_tokens, eos_token_ids))
        return out

    window = prompts[:args.batch_size]
    start_time = time.time()
    full = batched(window, args.batch_size)
    print(f"(one {args.batch_size}-row batch: {time.time() - start_time:.1f}s)")
    repeat = batched(window, args.batch_size)

    print("\n" + "=" * 72)
    print("Check C: is the batched path deterministic run-to-run?")
    unstable = [i for i in range(len(window)) if full[i] != repeat[i]]
    print(f"  {len(window) - len(unstable)}/{len(window)} identical on a repeat of the identical command")
    if unstable:
        print(f"  FAIL -- indices {[i + start + 1 for i in unstable]} changed between "
              f"identical runs; the eval is not reproducible")
        print("=" * 72)
        return 1
    print("  PASS -- batching is reproducible, which is what the matrix needs")

    singles = batched(window, 1)
    diff_vs_single = [i for i in range(len(window)) if full[i] != singles[i]]
    print("\n" + "=" * 72)
    print("Check D: 1-row batch (no left-padding) vs the full batch")
    print(f"  {len(window) - len(diff_vs_single)}/{len(window)} identical")
    for i in diff_vs_single:
        print(f"  index {i + start + 1}: 1-row {extract_final_answer(singles[i]['text'])}"
              f" vs {args.batch_size}-row {extract_final_answer(full[i]['text'])}")
    if diff_vs_single:
        print("  -> these differ with ZERO padding involved, so left-padding and the")
        print("     shared slice index are not what causes the disagreement")

    diff_vs_shift = []
    if len(prompts) > args.batch_size:
        shifted = batched(prompts[1:args.batch_size + 1], args.batch_size)
        shared = min(len(full), len(shifted))
        diff_vs_shift = [i for i in range(shared) if full[i] != shifted[i]]
        print("\n" + "=" * 72)
        print("Check E: same batch size, different neighbours (padding unchanged)")
        print(f"  {shared - len(diff_vs_shift)}/{shared} identical")
        for i in diff_vs_shift:
            print(f"  index {i + start + 1}: window 0 {extract_final_answer(full[i]['text'])}"
                  f" vs window 1 {extract_final_answer(shifted[i]['text'])}")
        if diff_vs_shift:
            print("  -> the SAME prompt with the SAME padding gives a different answer")
            print("     when only its batch-mates change: that is arithmetic order, not logic")
    else:
        print("\n(Check E skipped: pass --n_examples greater than --batch_size)")

    print("\n" + "=" * 72)
    print("Divergence detail (token level)")
    changed = sorted(set(diff_vs_single) | set(diff_vs_shift))
    if not changed:
        print("  none -- the two paths agreed everywhere in this window")
    for i in changed[:args.max_report]:
        neighbor = window[1] if i == 0 else window[0]
        single_seqs, scores = generate_token_ids(
            model, tokenizer, [window[i]], args.max_new_tokens, eos_token_ids
        )
        batched_seqs, _ = generate_token_ids(
            model, tokenizer, [neighbor, window[i]], args.max_new_tokens, eos_token_ids
        )
        d = first_divergence(single_seqs[0], batched_seqs[1])
        detail = ""
        if d is not None and d < len(scores):
            top2 = torch.topk(scores[d][0].float(), 2).values
            detail = (f"first differing token at step {d} "
                      f"({single_seqs[0][d]} vs {batched_seqs[1][d]}); "
                      f"top1-top2 logit gap there = {float(top2[0] - top2[1]):.4f}")
        else:
            detail = f"lengths differ: {len(single_seqs[0])} vs {len(batched_seqs[1])} tokens"
        print(f"  index {i + start + 1}: {detail}")
    print("  A gap near 0.0 is the signature of a near-tie that rounding order can flip;")
    print("  once one token flips, the two continuations are different sentences.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
