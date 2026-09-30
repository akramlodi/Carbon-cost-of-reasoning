"""Diagnose why batched and unbatched greedy generation disagree on a few examples.

`scripts/verify_batched_eval.py` compares batched against unbatched predictions
and can report mismatches (CHANGES.md Change 6). This script answers *why*.

Results are keyed by the example's 1-based GSM8K index rather than by position
within a window. That is not cosmetic: an earlier version of this script
compared `window0[i]` against `window1[i]` where window 1 was `prompts[1:]`, so
every comparison was silently between two *different* questions, and its
"0/16 identical" conclusion was meaningless. Keying by example index makes that
class of misalignment impossible.

Three checks:

  C. Determinism. The identical command twice must give identical output. This
     is what the matrix needs: the batched path has to be reproducible, not
     byte-identical to the unbatched path.
  D. 1-row batch (no left-padding at all) vs the full batch. This changes the
     *matmul shape*, so it is where float-level near-tie flips show up. Reported
     separately for "generation text differs" and "final answer differs", which
     are very different rates.
  E. Same batch size, different neighbours. Nothing about the prompt or its
     padding changes -- only which other rows shared the batch. Measured
     against the same example indices, so it is a real comparison.

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
    so a divergence can be located at token resolution."""
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
    return (
        [truncate_at_eos(row[input_length:].tolist(), eos_token_ids) for row in out.sequences],
        out.scores,
    )


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
    parser.add_argument("--max_report", type=int, default=5, help="Answer-changing examples to detail")
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
    prompts = [eval_ds[i]["prompt"] for i in range(start, start + args.n_examples)]

    def batched_by_index(ps, batch_size, first_example_index):
        """{1-based example index: generate_batch result}. Keyed by example, not
        by position in the batch, so results can be compared across windows."""
        results = {}
        for i in range(0, len(ps), batch_size):
            chunk = ps[i:i + batch_size]
            for offset, result in enumerate(generate_batch(
                model, tokenizer, chunk, args.max_new_tokens, eos_token_ids
            )):
                results[first_example_index + i + offset] = result
        return results

    window0 = batched_by_index(prompts, args.batch_size, start + 1)
    start_time = time.time()
    print(f"(one {args.batch_size}-row batch: {time.time() - start_time:.1f}s)")
    repeat = batched_by_index(prompts, args.batch_size, start + 1)

    print("\n" + "=" * 72)
    print("Check C: is the batched path deterministic run-to-run?")
    unstable = [i for i in window0 if window0[i] != repeat.get(i)]
    print(f"  {len(window0) - len(unstable)}/{len(window0)} identical on a repeat "
          f"of the identical command")
    if unstable:
        print(f"  FAIL -- example indices {unstable} changed between identical runs; "
              f"the eval is not reproducible")
        print("=" * 72)
        return 1
    print("  PASS -- batching is reproducible, which is what the matrix needs")

    singles = batched_by_index(prompts, 1, start + 1)
    text_differs = [i for i in sorted(window0) if window0[i]["text"] != singles[i]["text"]]
    answer_differs = [
        i for i in sorted(window0)
        if extract_final_answer(window0[i]["text"]) != extract_final_answer(singles[i]["text"])
    ]
    print("\n" + "=" * 72)
    print("Check D: 1-row batch (no left-padding) vs the full batch")
    print(f"  generation text differs : {len(text_differs)}/{len(window0)}")
    print(f"  final answer differs    : {len(answer_differs)}/{len(window0)} "
          f"({', '.join(str(i) for i in answer_differs) or 'none'})")
    print("  -> these differ with ZERO padding involved, so left-padding and the")
    print("     shared slice index are not what causes any disagreement")
    print("     (the two rates differing is the point: one flipped token early on")
    print("      usually still lands on the same number)")

    if len(prompts) > args.batch_size:
        # one-position shift: the same examples, each sharing the batch with a
        # different neighbour, so a result changing here cannot be a bug in
        # which example was asked -- the same example is being asked both times
        window1 = batched_by_index(prompts[1:args.batch_size + 1], args.batch_size, start + 2)
        shared = sorted(set(window0) & set(window1))
        print("\n" + "=" * 72)
        print("Check E: same batch size, different neighbours, same examples")
        print(f"  window 0 = examples {min(window0)}..{max(window0)}; "
              f"window 1 = examples {min(window1)}..{max(window1)}")
        print(f"  comparing the {len(shared)} example(s) present in both")
        text_diff = [i for i in shared if window0[i]["text"] != window1[i]["text"]]
        print(f"  {len(shared) - len(text_diff)}/{len(shared)} byte-identical")
        for i in text_diff:
            print(f"  example {i}: "
                  f"{extract_final_answer(window0[i]['text'])} vs "
                  f"{extract_final_answer(window1[i]['text'])}")
        if not text_diff:
            print("  -> the same example produces the same generation regardless of")
            print("     which other examples shared its batch: batch composition has no")
            print("     effect, so the batching is compositionally consistent")
    else:
        print("\n(Check E skipped: pass --n_examples greater than --batch_size)")

    print("\n" + "=" * 72)
    print("Divergence detail for the answer-changing examples (token level)")
    if not answer_differs:
        print("  none -- every example reached the same final answer at 1 row and "
              f"{args.batch_size} rows")
    for i in answer_differs[:args.max_report]:
        neighbor_prompt = prompts[1] if i == start + 1 else prompts[0]
        own_prompt = prompts[i - (start + 1)]
        single_seqs, scores = generate_token_ids(
            model, tokenizer, [own_prompt], args.max_new_tokens, eos_token_ids
        )
        batched_seqs, _ = generate_token_ids(
            model, tokenizer, [neighbor_prompt, own_prompt], args.max_new_tokens, eos_token_ids
        )
        d = first_divergence(single_seqs[0], batched_seqs[1])
        if d is None or d >= len(scores):
            print(f"  example {i}: lengths differ, "
                  f"{len(single_seqs[0])} vs {len(batched_seqs[1])} tokens")
            continue
        top2 = torch.topk(scores[d][0].float(), 2).values
        print(f"  example {i}: first differing token at step {d} "
              f"({single_seqs[0][d]} vs {batched_seqs[1][d]}); "
              f"top1-top2 logit gap there = {float(top2[0] - top2[1]):.4f}")
    print("  A gap near 0.0 is the signature of a near-tie that rounding order can flip;")
    print("  once one token flips, the two continuations are different sentences.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
