"""Prints raw model generations for a handful of GSM8K examples --
qualitative inspection to complement measure_baseline.py's aggregate
accuracy number, which only ever shows the extracted pred/ref pair.

Useful for exactly the kind of thing an aggregate number hides: does the
constructed prompt look right (no duplicated BOS, sane turn markers -- see
the add_special_tokens=False fix in src/data.py and src/evaluate.py), does
the model stop generating cleanly, and does the reasoning in the raw text
actually look sane before trusting the extracted number at all.

Usage:
    python scripts/inspect_generations.py --n_examples 5
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
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=256):
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
        "prompt_as_sent": tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False),
        "generated_clean": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_raw": tokenizer.decode(generated_ids, skip_special_tokens=False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    print(f"Loading model ({MODEL_ID}, bf16, no LoRA, no quantization)...")
    config = {"model_id": MODEL_ID, "bf16": True}
    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)
    model.eval()

    eval_ds = load_gsm8k(tokenizer, args.split, n_samples=args.n_examples)

    for i, example in enumerate(eval_ds, start=1):
        reference = extract_final_answer(example["answer"])
        result = generate_and_decode(
            model, tokenizer, example["prompt"], max_new_tokens=args.max_new_tokens
        )
        predicted = extract_final_answer(result["generated_clean"])
        correct = answers_match(predicted, reference)

        print(f"\n{'=' * 80}")
        print(f"[{i}] correct={correct}  pred={predicted}  ref={reference}")
        print(f"{'-' * 80}")
        print(f"QUESTION:\n{example['question']}")
        print(f"{'-' * 80}")
        print("PROMPT AS SENT TO MODEL (special tokens visible -- check for")
        print("duplicated BOS or malformed turn markers):")
        print(result["prompt_as_sent"])
        print(f"{'-' * 80}")
        print("RAW GENERATED TEXT (special tokens visible):")
        print(result["generated_raw"])
        print(f"{'-' * 80}")
        print("CLEAN GENERATED TEXT (skip_special_tokens=True -- this is what gets scored):")
        print(result["generated_clean"])

    print(f"\n{'=' * 80}")
    print(f"Done -- inspected {len(eval_ds)} examples from the {args.split} split.")


if __name__ == "__main__":
    main()
