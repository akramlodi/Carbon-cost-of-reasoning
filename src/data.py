"""GSM8K loading, prompt formatting, and tokenization -- shared across all
three fine-tuning conditions so the training/eval distribution is identical.

Prompts go through the tokenizer's chat template rather than a raw
completion-style string: google/gemma-4-E2B-it is instruction-tuned and
expects its own turn-formatted input (special tokens, role markers) to
perform as intended -- a bare "Question: ...\\nAnswer: ..." string is what
its base (non-"-it") counterpart would want, not this checkpoint.
"""
from datasets import load_dataset

INSTRUCTION = (
    "Question: {question}\n"
    "Answer: Let's think step by step. End your response with only: #### <number>"
)


def _user_turn(question):
    return {"role": "user", "content": INSTRUCTION.format(question=question)}


def format_example(example, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [_user_turn(example["question"])], tokenize=False, add_generation_prompt=True
    )
    text = tokenizer.apply_chat_template(
        [_user_turn(example["question"]), {"role": "assistant", "content": example["answer"]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"prompt": prompt, "text": text}


def load_gsm8k(tokenizer, split="train", n_samples=None):
    """split: 'train' or 'test'. n_samples slices the split for smoke tests."""
    split_expr = f"{split}[:{n_samples}]" if n_samples else split
    ds = load_dataset("openai/gsm8k", "main", split=split_expr)
    return ds.map(lambda example: format_example(example, tokenizer))


def tokenize_dataset(ds, tokenizer, max_length=512):
    """Truncates to max_length but does not pad -- padding is left to a
    per-batch dynamic collator (see src.train.build_training_args's caller,
    which wires up DataCollatorForLanguageModeling) so short examples don't
    pay for a full max_length forward pass. That matters here specifically:
    on a memory-constrained GPU, padding every example out to max_length
    inflates the [batch, seq_len, vocab_size] logits tensor for the loss
    computation regardless of actual content length, and gemma-4's ~262k
    vocab makes that tensor large enough to be the difference between
    fitting and OOMing.

    add_special_tokens=False: ds["text"] already went through the
    tokenizer's chat template (see format_example), which embeds BOS/turn
    tokens itself -- re-adding them here would duplicate BOS.
    """
    def _tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length, add_special_tokens=False
        )

    return ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
