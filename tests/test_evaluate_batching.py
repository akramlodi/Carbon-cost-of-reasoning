"""Tests for the batched generation path in src/evaluate.py.

Two layers, because they catch different things:

1. A stub model whose `generate()` output we control exactly, to pin down the
   semantics the batching depends on -- the shared input-length slice under
   left-padding, per-row EOS detection (rows that stop early are still padded
   out to max_new_tokens in the output tensor, so a shared cutoff would be
   wrong), multi-token EOS sets, row ordering, and that generation really does
   run under torch.no_grad (the mid-training memory leak).

2. A tiny real HF causal LM on the project's real tokenizer, to check that
   batched generation reproduces the unbatched `generate_answer()` reference
   path example-for-example. This is the same question the GPU check asks of
   the real gemma-4 checkpoint, at a size that runs on CPU.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import (
    EVAL_BATCH_SIZE,
    MID_TRAIN_EVAL_BATCH_SIZE,
    evaluate_model,
    generate_answer,
    generate_batch,
    normalize_eos_ids,
)

MODEL_ID = "google/gemma-4-E2B-it"
PAD_ID = 0
EOS_IDS = {1, 106}  # gemma's <eos> and <end_of_turn>


class StubCausalLM:
    """Minimal stand-in for the surface src/evaluate.generate_batch touches:
    .device / .training / .eval() / .train() / .generation_config / .generate().

    `continuations` maps a row's first real prompt token to the token sequence
    that row is scripted to emit. Keying off the prompt rather than the row's
    position in the batch is deliberate: it means a row gets the same
    generation no matter which batch it lands in, so evaluate_model pairing
    each example back to its own generation is actually under test.

    `generate()` reproduces what real HF generate() hands back -- the full
    left-padded prompt, then exactly max_new_tokens columns, with rows whose
    scripted continuation already emitted an EOS padded out to the end.
    """

    def __init__(self, continuations):
        self.continuations = continuations
        self.device = torch.device("cpu")
        self.training = True
        self.grad_enabled_during_generate = []
        self.pad_prefix_lengths = []
        self.generation_config = type("GenCfg", (), {"eos_token_id": sorted(EOS_IDS)})()

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8,
                 do_sample=False, pad_token_id=PAD_ID, **kwargs):
        self.grad_enabled_during_generate.append(torch.is_grad_enabled())
        input_length = input_ids.shape[1]
        assert attention_mask.shape == input_ids.shape
        # left-padding invariant: the mask is a 0-prefix then all 1s and every
        # row's real content ends at the same absolute index
        self.pad_prefix_lengths = []
        for row, mask_row in zip(input_ids.tolist(), attention_mask.tolist()):
            assert mask_row == sorted(mask_row), "mask must be a 0-prefix then all 1s"
            start = mask_row.index(1)
            assert start + sum(mask_row) == input_length
            assert all(t == PAD_ID for t in row[:start]), "pad prefix must be pad tokens"
            self.pad_prefix_lengths.append(start)

        rows = []
        for prompt, mask_row in zip(input_ids.tolist(), attention_mask.tolist()):
            first_real_token = prompt[mask_row.index(1)]
            gen = list(self.continuations[first_real_token])[:max_new_tokens]
            if any(t in EOS_IDS for t in gen):
                # real generate() abandons the rest of a finished row and
                # fills the remaining columns with pad tokens
                gen = gen[:next(j for j, t in enumerate(gen) if t in EOS_IDS) + 1]
            rows.append(prompt + gen + [pad_token_id] * (max_new_tokens - len(gen)))
        return torch.tensor(rows, dtype=input_ids.dtype)


class StubTokenizer:
    """Whitespace tokenizer with a real pad token, so left-padding actually
    happens: the per-row prompt lengths below differ on purpose."""

    pad_token = "<pad>"
    pad_token_id = PAD_ID
    eos_token_id = 1

    def __init__(self):
        self.padding_side = "right"  # deliberately wrong, generate_batch must set its own
        self._vocab = {}

    def id_of(self, word):
        return self._vocab.setdefault(word, 1000 + len(self._vocab))

    def __call__(self, prompts, return_tensors=None, padding=False, add_special_tokens=False):
        rows = [[self.id_of(w) for w in p.split()] for p in prompts]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for row in rows:  # left-padding
            pad = width - len(row)
            ids.append([self.pad_token_id] * pad + row)
            mask.append([0] * pad + [1] * len(row))
        batch = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }
        return _BatchEncoding(batch)

    def decode(self, ids, skip_special_tokens=False):
        specials = {self.pad_token_id, self.eos_token_id, 106}
        tokens = [str(t) for t in ids if not (skip_special_tokens and t in specials)]
        return " ".join(tokens)


class _BatchEncoding(dict):
    def to(self, device):
        return self


def make_dataset(prompt_word_counts, references=None):
    return [
        {
            "question": f"question {i}",
            "prompt": " ".join(f"p{i}w{j}" for j in range(prompt_word_counts[i])),
            "answer": f"work ... #### {references[i] if references else i * 2}",
        }
        for i in range(len(prompt_word_counts))
    ]


def script_continuations(tokenizer, continuations):
    """Key a continuation list by the first real token of each prompt, so the
    stub returns the same generation for a prompt wherever it appears."""
    return {
        tokenizer.id_of(f"p{i}w0"): continuation for i, continuation in enumerate(continuations)
    }


def test_generate_batch_recovers_per_row_eos_length_and_stops_at_first_eos():
    continuations = [
        [11, 12, 1],           # emits <eos> third -> 3 generated tokens
        [21, 1, 22, 23],       # emits <eos> second -> 2, and 22/23 never happened
        [31, 106],             # <end_of_turn> is a valid stop too -> 2 tokens
        [41, 42, 43, 44],      # never emits eos -> still generating at the cap
    ]
    tokenizer = StubTokenizer()
    model = StubCausalLM(script_continuations(tokenizer, continuations))
    # uneven prompt lengths (2, 5, 3, 4 words) so left-padding is non-trivial
    prompts = ["p0w0 p0w1", "p1w0 p1w1 p1w2 p1w3 p1w4", "p2w0 p2w1 p2w2", "p3w0 p3w1 p3w2 p3w3"]

    results = generate_batch(model, tokenizer, prompts, max_new_tokens=6, eos_token_ids=EOS_IDS)

    # prompts are 2/5/3/4 tokens: the shorter ones must be pad-prefixed, which
    # is what makes a single shared slice index legal
    assert model.pad_prefix_lengths == [3, 0, 2, 1]
    assert [r["n_generated_tokens"] for r in results] == [3, 2, 2, 6]
    assert [r["hit_token_limit"] for r in results] == [False, False, False, True]
    # the shared input_length slice must isolate exactly the continuation:
    # no leading pad tokens, no prompt tokens leaking in
    assert [r["text"] for r in results] == ["11 12", "21", "31", "41 42 43 44"]


def test_generate_batch_preserves_prompt_order_and_restores_padding_side():
    tokenizer = StubTokenizer()
    model = StubCausalLM(script_continuations(tokenizer, [[10, 10], [11, 11], [12, 12]]))
    assert tokenizer.padding_side == "right"

    prompts = ["p0w0", "p1w0 p1w1 p1w2", "p2w0 p2w1"]
    results = generate_batch(model, tokenizer, prompts, max_new_tokens=4, eos_token_ids=EOS_IDS)

    assert [r["text"] for r in results] == ["10 10", "11 11", "12 12"]
    # the shared tokenizer is also the trainer's collator -- its padding side
    # must come back untouched
    assert tokenizer.padding_side == "right"


def test_generate_batch_runs_without_grad():
    tokenizer = StubTokenizer()
    model = StubCausalLM(script_continuations(tokenizer, [[1], [2]]))
    generate_batch(model, tokenizer, ["p0w0", "p1w0 p1w1"], max_new_tokens=3, eos_token_ids=EOS_IDS)

    assert model.grad_enabled_during_generate == [False]


def test_normalize_eos_ids_handles_int_list_and_missing():
    model = StubCausalLM({})
    assert normalize_eos_ids(model, StubTokenizer()) == EOS_IDS

    model.generation_config.eos_token_id = 7
    assert normalize_eos_ids(model, StubTokenizer()) == {7}

    model.generation_config.eos_token_id = None
    assert normalize_eos_ids(model, StubTokenizer()) == {StubTokenizer.eos_token_id}


def test_same_prompt_generates_identically_whatever_its_batch_mates_are():
    """Compositional consistency: a prompt's generation must not depend on which
    other rows shared its batch.

    This is the invariant scripts/diagnose_batched_divergence.py Check E is
    supposed to measure on the real checkpoint. An earlier version of that
    script compared `window0[i]` against `window1[i]` where window 1 was
    `prompts[1:]`, so every comparison was between two *different* prompts and
    its conclusion was meaningless -- hence pinning it down here on CPU, where
    the stub's generations are known exactly.
    """
    tokenizer = StubTokenizer()
    continuations = [[11, 12], [21, 22], [31, 32]]
    model = StubCausalLM(script_continuations(tokenizer, continuations))
    prompts = ["p0w0 p0w1", "p1w0 p1w1 p1w2 p1w3", "p2w0 p2w1 p2w2"]

    together = generate_batch(model, tokenizer, prompts, max_new_tokens=4, eos_token_ids=EOS_IDS)
    alone = [generate_batch(model, tokenizer, [p], max_new_tokens=4, eos_token_ids=EOS_IDS)[0]
             for p in prompts]

    assert [r["text"] for r in together] == ["11 12", "21 22", "31 32"]
    assert together == alone


def test_evaluate_model_scores_batched_generation_and_restores_train_mode():
    tokenizer = StubTokenizer()
    model = StubCausalLM(script_continuations(tokenizer, [[11, 12], [21, 22], [31, 32]]))
    # extract_final_answer takes the last number in the generation, so the
    # references have to line up with 12/22/32
    dataset = make_dataset([2, 4, 3], references=[99, 22, 98])

    accuracy, records = evaluate_model(
        model, tokenizer, dataset, max_new_tokens=4, batch_size=2, log_every_n_batches=0
    )

    # predictions are 12/22/32 -> only the middle one matches its reference,
    # including the one in the leftover single-example second batch
    assert accuracy == pytest.approx(1 / 3)
    assert [r["predicted"] for r in records] == [12.0, 22.0, 32.0]
    assert [r["reference"] for r in records] == [99.0, 22.0, 98.0]
    assert [r["correct"] for r in records] == [False, True, False]
    assert [r["question"] for r in records] == [f"question {i}" for i in range(3)]
    # evaluate_model is called mid-training by GreenGapCheckpointCallback, so
    # it has to hand the model back in training mode
    assert model.training is True


def test_batch_size_defaults_differ_by_call_site():
    # the mid-training eval has to fit alongside live optimizer state, so it
    # must not silently inherit the standalone eval's batch size
    assert MID_TRAIN_EVAL_BATCH_SIZE < EVAL_BATCH_SIZE


def _load_real_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tiny_causal_lm(vocab_size):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, eos_token_id=1, pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    # Amplify the output head so greedy argmax gaps dwarf the float
    # differences left-padding introduces; without this, a random-init model
    # can flip a near-tie and diverge, which would make the parity check
    # below flaky rather than informative.
    with torch.no_grad():
        model.lm_head.weight.mul_(50.0)
    return model


@pytest.fixture(scope="module")
def real_tokenizer():
    try:
        return _load_real_tokenizer()
    except Exception as exc:  # no local HF cache / no network
        pytest.skip(f"could not load {MODEL_ID}'s tokenizer: {exc}")


def test_batched_generation_matches_single_example_generation(real_tokenizer):
    tokenizer = real_tokenizer
    model = _tiny_causal_lm(len(tokenizer))

    # GSM8K-style prompts of deliberately uneven length, so rows in a batch
    # have different amounts of left-padding
    lengths = [12, 40, 25, 8, 33, 17]
    dataset = [
        {
            "question": f"q{i}",
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": " ".join(["word"] * length)}],
                tokenize=False, add_generation_prompt=True,
            ),
            "answer": f"reasoning #### {i}",
        }
        for i, length in enumerate(lengths)
    ]
    max_new_tokens = 12
    eos_token_ids = normalize_eos_ids(model, tokenizer)

    unbatched = [generate_answer(model, tokenizer, ex["prompt"], max_new_tokens) for ex in dataset]
    batched = generate_batch(
        model, tokenizer, [ex["prompt"] for ex in dataset], max_new_tokens, eos_token_ids
    )

    assert [r["text"] for r in batched] == unbatched
    assert all(r["n_generated_tokens"] == max_new_tokens for r in batched)


def test_evaluate_model_batched_matches_unbatched_predictions(real_tokenizer):
    tokenizer = real_tokenizer
    model = _tiny_causal_lm(len(tokenizer))
    dataset = [
        {
            "question": f"q{i}",
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": " ".join(["word"] * length)}],
                tokenize=False, add_generation_prompt=True,
            ),
            "answer": f"reasoning #### {i}",
        }
        for i, length in enumerate([12, 40, 25, 8, 33, 17])
    ]
    max_new_tokens = 12

    batched_accuracy, batched_records = evaluate_model(
        model, tokenizer, dataset, max_new_tokens=max_new_tokens, batch_size=3,
        log_every_n_batches=0,
    )

    # the reference path: one example at a time, no batching at all
    def unbatched(dataset, max_new_tokens=768, **kwargs):
        from src.evaluate import answers_match, extract_final_answer

        records = []
        for example in dataset:
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
        accuracy = sum(r["correct"] for r in records) / len(records)
        return accuracy, records

    unbatched_accuracy, unbatched_records = unbatched(dataset, max_new_tokens)

    assert batched_records == unbatched_records
    assert batched_accuracy == pytest.approx(unbatched_accuracy)
