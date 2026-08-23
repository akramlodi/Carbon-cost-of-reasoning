"""Pulls specific GSM8K test-set examples (by their 1-based position in the
same slice measure_baseline.py evaluated) through the raw-generation
diagnostic, to check whether a wrong answer is a genuine reasoning failure
or a truncation artifact -- generation hitting max_new_tokens before ever
emitting the "#### <number>" marker.

Companion to scripts/inspect_generations.py, which only supports "first N";
this one targets specific example numbers, so you can go straight to the
wrong answers a measure_baseline.py run flagged instead of hoping they show
up in an arbitrary first-N sample.

For each requested index, reports a computed (not eyeballed) verdict: did
generation stop naturally (EOS reached before the token budget), or did it
hit max_new_tokens -- the latter means the true prediction is unknown, not
necessarily wrong, since the model may have been about to state the answer.

Usage -- indices are 1-based, matching measure_baseline.py's "[N] correct=..."
output, against the SAME --n_eval_samples slice that produced it:
    python scripts/inspect_by_index.py --indices 9,13,14,15,40,42 --n_eval_samples 50

Match --max_new_tokens to whatever produced the run you're auditing
(measure_baseline.py currently defaults to 256, not the 512 used by
inspect_generations.py -- check which one actually produced the numbers
you're looking at before concluding anything).
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import load_gsm8k
from src.evaluate import answers_match, extract_final_answer
from src.models import load_model, load_tokenizer

MODEL_ID = "google/gemma-4-E2B-it"


@torch.no_grad()
def generate_and_decode(model, tokenizer, prompt, max_new_tokens):
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
        "n_generated": generated_ids.shape[0],
        "hit_token_limit": generated_ids.shape[0] >= max_new_tokens,
        "generated_clean": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_raw": tokenizer.decode(generated_ids, skip_special_tokens=False),
    }


def parse_indices(raw):
    return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", required=True, help="Comma-separated 1-based positions, e.g. 9,13,14,15,40,42")
    parser.add_argument("--n_eval_samples", type=int, default=50, help="Must match the slice size the run you're auditing used")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Must match the run you're auditing (measure_baseline.py defaults to 256)")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    indices = parse_indices(args.indices)
    if any(i < 1 or i > args.n_eval_samples for i in indices):
        raise ValueError(f"--indices must be within 1..{args.n_eval_samples} (your --n_eval_samples)")

    print(f"Loading model ({MODEL_ID}, bf16, no LoRA, no quantization)...")
    config = {"model_id": MODEL_ID, "bf16": True}
    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)
    model.eval()

    eval_ds = load_gsm8k(tokenizer, args.split, n_samples=args.n_eval_samples)

    truncated_count = 0
    for idx in indices:
        example = eval_ds[idx - 1]
        reference = extract_final_answer(example["answer"])
        result = generate_and_decode(
            model, tokenizer, example["prompt"], max_new_tokens=args.max_new_tokens
        )
        predicted = extract_final_answer(result["generated_clean"])
        correct = answers_match(predicted, reference)
        verdict = "TRUNCATED (hit max_new_tokens -- true prediction unknown)" if result["hit_token_limit"] else f"stopped naturally ({result['n_generated']} tokens)"
        if result["hit_token_limit"]:
            truncated_count += 1

        print(f"\n{'=' * 80}")
        print(f"[{idx}] correct={correct}  pred={predicted}  ref={reference}  -- {verdict}")
        print(f"{'-' * 80}")
        print(f"QUESTION:\n{example['question']}")
        print(f"{'-' * 80}")
        print(f"REFERENCE SOLUTION:\n{example['answer']}")
        print(f"{'-' * 80}")
        print("RAW GENERATED TEXT (special tokens visible):")
        print(result["generated_raw"])
        print(f"{'-' * 80}")
        print("CLEAN GENERATED TEXT (skip_special_tokens=True -- this is what gets scored):")
        print(result["generated_clean"])

    print(f"\n{'=' * 80}")
    print(f"Done -- {len(indices)} examples checked, {truncated_count} hit the token limit before stopping naturally.")
    if truncated_count:
        print(f"Those {truncated_count} predictions are unreliable regardless of whether they scored correct/incorrect -- generation was cut off, not concluded.")


if __name__ == "__main__":
    main()
