"""GSM8K loading, prompt formatting, and tokenization -- shared across all
three fine-tuning conditions so the training/eval distribution is identical.
"""
from datasets import load_dataset

PROMPT_TEMPLATE = "Question: {question}\nAnswer: Let's think step by step.\n"


def format_example(example):
    prompt = PROMPT_TEMPLATE.format(question=example["question"])
    return {
        "prompt": prompt,
        "text": prompt + example["answer"],
    }


def load_gsm8k(split="train", n_samples=None):
    """split: 'train' or 'test'. n_samples slices the split for smoke tests."""
    split_expr = f"{split}[:{n_samples}]" if n_samples else split
    ds = load_dataset("openai/gsm8k", "main", split=split_expr)
    return ds.map(format_example)


def tokenize_dataset(ds, tokenizer, max_length=512):
    """Truncates to max_length but does not pad -- padding is left to a
    per-batch dynamic collator (see src.train.build_data_collator) so short
    examples don't pay for a full max_length forward pass. That matters here
    specifically: on a memory-constrained GPU, padding every example out to
    max_length inflates the [batch, seq_len, vocab_size] logits tensor for
    the loss computation regardless of actual content length, and gemma-4's
    ~262k vocab makes that tensor large enough to be the difference between
    fitting and OOMing.
    """
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
