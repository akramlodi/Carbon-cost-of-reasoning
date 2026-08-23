"""GSM8K exact-match scoring and numeric answer extraction.

Answers are compared numerically (not as strings) so formatting differences
like `12` vs `12.0` don't count as mismatches.
"""
import re

import torch

ANSWER_MARKER = "####"
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_answer(text):
    """Pull the last number after the #### marker if present, else the last
    number in the text at all (predictions won't always emit the marker)."""
    if ANSWER_MARKER in text:
        text = text.split(ANSWER_MARKER)[-1]
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def answers_match(pred, ref, tol=1e-4):
    if pred is None or ref is None:
        return False
    return abs(pred - ref) < tol


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    # add_special_tokens=False: prompt already went through the tokenizer's
    # chat template (see src.data.format_example), which embeds BOS/turn
    # tokens itself -- re-adding them here would duplicate BOS.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_model(model, tokenizer, dataset, max_new_tokens=256, verbose=False):
    """Returns (accuracy, per-example records). Temporarily switches the
    model to eval mode and restores the prior mode on exit, so this is safe
    to call mid-training from a checkpoint callback."""
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    records = []
    try:
        for example in dataset:
            prompt = example["prompt"]
            reference = extract_final_answer(example["answer"])
            generated = generate_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            predicted = extract_final_answer(generated)
            is_correct = answers_match(predicted, reference)

            correct += int(is_correct)
            total += 1
            records.append({
                "question": example["question"],
                "reference": reference,
                "predicted": predicted,
                "correct": is_correct,
            })
            if verbose:
                print(f"[{total}] correct={is_correct} pred={predicted} ref={reference}")
    finally:
        if was_training:
            model.train()

    accuracy = correct / total if total else 0.0
    return accuracy, records
