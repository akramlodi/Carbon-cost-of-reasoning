"""GSM8K exact-match scoring and numeric answer extraction.

Answers are compared numerically (not as strings) so formatting differences
like `12` vs `12.0` don't count as mismatches.

Generation is batched (left-padded, decoder-only-correct) rather than one
example at a time. The trailing eval at the end of every src/train.py run
scores the full 1,319-example GSM8K test set, and at the observed ~1
min/example that is well over an hour of ~30%-utilized GPU -- paid again by
every run in the matrix. scripts/measure_baseline.py hit the identical wall
measuring the zero-shot baseline and solved it the same way; this module is
now the single implementation of that logic and measure_baseline.py imports
it from here, so the two can't drift apart.
"""
import re
import time

import torch

ANSWER_MARKER = "####"
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# Batch size for a standalone eval (nothing else on the GPU). Same default as
# scripts/measure_baseline.py's --batch_size, which was sized to fill a g5.xlarge
# A10G's 24GB during baseline measurement. Back off to 8 or 4 if this OOMs.
EVAL_BATCH_SIZE = 16

# Batch size for the mid-training eval in src/train.py's
# GreenGapCheckpointCallback -- deliberately smaller than EVAL_BATCH_SIZE,
# because that call site is not standalone: it runs against a model that
# still holds its optimizer state and resumes training immediately after,
# on a card where a 262,144-vocab logits tensor plus its softcapping copy
# already OOM'd at per_device_train_batch_size=8 (CHANGES.md Change 5).
# Generation under no_grad doesn't build an activation graph, so the extra
# cost is mostly the KV cache, but there's no reason to hand a live
# training process a 16-row cache it has no headroom for.
MID_TRAIN_EVAL_BATCH_SIZE = 4


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
def generate_answer(model, tokenizer, prompt, max_new_tokens=768):
    """One prompt, one generation, no batching -- too slow for the 1,319-example
    end-of-run eval (see module docstring), but kept as the reference path the
    batched generate_batch is checked against in tests/test_evaluate_batching.py."""
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
def generate_batch(model, tokenizer, prompts, max_new_tokens=768, eos_token_ids=None):
    """Batched generation with left-padding. Returns one result dict per
    prompt, in the order given: text / n_generated_tokens / hit_token_limit.

    Left-padding means every row's real prompt content ends at the same
    absolute position (input_ids.shape[1]) regardless of how much padding
    precedes it -- that's the whole mechanism that makes batched
    decoder-only generation possible, and it's why a single shared slice
    index (input_length) correctly isolates the generated continuation for
    every row; right-padding would make that slice point vary per row and
    silently produce garbage. padding_side is set here and restored on exit
    rather than at the call site: the training path shares this tokenizer
    with the trainer's data collator, and flipping its padding side
    permanently would left-pad training batches too.

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
    if eos_token_ids is None:
        eos_token_ids = normalize_eos_ids(model, tokenizer)

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
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
    finally:
        tokenizer.padding_side = previous_padding_side

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


@torch.no_grad()
def evaluate_model(
    model, tokenizer, dataset, max_new_tokens=768, batch_size=EVAL_BATCH_SIZE,
    log_every_n_batches=10, verbose=False,
):
    """Returns (accuracy, per-example records). Temporarily switches the
    model to eval mode and restores the prior mode on exit, so this is safe
    to call mid-training from a checkpoint callback.

    batch_size is a parameter rather than a single hardcoded default because
    the two call sites want different things: the end-of-run eval in
    src/train.py::run is standalone and wants EVAL_BATCH_SIZE, while the
    mid-training call from GreenGapCheckpointCallback is competing with a
    live optimizer state and uses the smaller MID_TRAIN_EVAL_BATCH_SIZE.
    log_every_n_batches prints a progress line (0 disables) -- a 1,319-example
    eval otherwise runs silently for a long time inside tmux.
    """
    was_training = model.training
    model.eval()

    eos_token_ids = normalize_eos_ids(model, tokenizer)
    n = len(dataset)
    correct = 0
    total = 0
    records = []
    start_time = time.time()
    try:
        for batch_start in range(0, n, batch_size):
            examples = [dataset[i] for i in range(batch_start, min(batch_start + batch_size, n))]
            prompts = [example["prompt"] for example in examples]

            batch_results = generate_batch(
                model, tokenizer, prompts, max_new_tokens, eos_token_ids
            )

            for example, gen in zip(examples, batch_results):
                reference = extract_final_answer(example["answer"])
                predicted = extract_final_answer(gen["text"])
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

            batch_num = batch_start // batch_size + 1
            if log_every_n_batches and (
                batch_num % log_every_n_batches == 0 or total >= n
            ):
                elapsed = time.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0.0
                remaining = (n - total) / rate if rate > 0 else float("inf")
                print(
                    f"-- eval progress: {total}/{n} ({rate:.2f} ex/sec, "
                    f"batch_size={batch_size}, ETA {remaining / 60:.1f} min) --"
                )
    finally:
        if was_training:
            model.train()

    accuracy = correct / total if total else 0.0
    return accuracy, records
