"""Tiny end-to-end pipeline check, meant to run on free Colab (T4).

Exercises the QLoRA and LoRA paths only -- full fine-tuning needs more memory
than a T4 has (an OOM there is expected, not a bug; see README). This is not
a real experiment: it uses a tiny data subset and a handful of steps purely
to catch wiring bugs (data formatting, OOM on LoRA/QLoRA configs, tokenizer
mismatches, CodeCarbon init failures, checkpoint save/reload) before any
money is spent on a rented GPU.
"""
import argparse
import gc
import os
import sys

# Must be set before torch initializes CUDA (i.e. before any of the imports
# below, which pull torch in transitively) to have any effect. Free-tier T4
# runs this pipeline within a few hundred MB of its 15GB ceiling, so
# fragmentation-driven OOMs are a real risk even when the true peak usage
# would otherwise fit.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.data import load_gsm8k, tokenize_dataset
from src.energy_tracker import EnergyRun
from src.evaluate import evaluate_model, extract_final_answer
from src.models import load_model, load_tokenizer

MODEL_ID = "google/gemma-4-E2B-it"
N_SMOKE_SAMPLES = 32
N_SMOKE_STEPS = 10
N_EVAL_SAMPLES = 8
# Deliberately short: this is a wiring check, not a real run, and gemma-4's
# ~262k vocab makes the [batch, seq_len, vocab_size] loss logits tensor the
# single biggest consumer of headroom on a 15GB T4 -- keeping seq_len small
# here (on top of dynamic per-batch padding, see src/data.py) is what makes
# the smoke test fit at all.
SMOKE_MAX_LENGTH = 256


def build_smoke_config(method):
    return {
        "model_id": MODEL_ID,
        "bf16": True,
        "gradient_checkpointing": False,
        "quantization": {"enabled": method == "qlora"},
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            # no target_modules -- let peft>=0.19's Gemma-4-aware defaults
            # pick the right modules; see src/models.py::load_model.
        },
    }


def run_smoke_test(method="qlora"):
    assert method in ("qlora", "lora"), "smoke test only covers qlora/lora (full_ft OOMs on T4 by design)"
    config = build_smoke_config(method)

    print(f"[1/6] Loading tokenizer + {method} model ({MODEL_ID})...")
    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)

    print("[2/6] Loading + tokenizing a tiny GSM8K subset...")
    train_ds = load_gsm8k(tokenizer, "train", n_samples=N_SMOKE_SAMPLES)
    eval_ds = load_gsm8k(tokenizer, "test", n_samples=N_EVAL_SAMPLES)
    tokenized_train = tokenize_dataset(train_ds, tokenizer, max_length=SMOKE_MAX_LENGTH)

    print("[3/6] Starting energy tracker...")
    energy = EnergyRun(project_name="green-gap-smoke-test", output_dir="results/emissions")

    training_args = TrainingArguments(
        output_dir="smoke_test_output",
        per_device_train_batch_size=1,
        max_steps=N_SMOKE_STEPS,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_train, data_collator=data_collator
    )

    print(f"[4/6] Running {N_SMOKE_STEPS} training steps...")
    with energy:
        trainer.train()
    print(f"Smoke test training complete. Estimated emissions: {energy.emissions_kg} kg CO2e")

    print("[5/6] Sanity-checking answer extraction + evaluation...")
    assert extract_final_answer("blah blah #### 42") == 42.0
    assert extract_final_answer("no marker here, answer is 7") == 7.0
    accuracy, _ = evaluate_model(model, tokenizer, eval_ds, max_new_tokens=64)
    print(f"Smoke-test accuracy on {N_EVAL_SAMPLES} examples (not meaningful, just a wiring check): {accuracy:.2f}")

    print("[6/6] Sanity-checking adapter save/reload...")
    adapter_dir = "smoke_test_adapter"
    model.save_pretrained(adapter_dir)

    # Free the first model before loading a second copy -- otherwise
    # device_map="auto" has to offload part of the reload onto CPU/disk to
    # fit both copies at once, and peft has a bug loading an adapter onto a
    # partially-offloaded base model (KeyError on a *_norm submodule during
    # its offload-index bookkeeping). Loading straight onto a freshly-freed
    # GPU sidesteps that entirely.
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reload_base = load_model({**config, "lora": {"enabled": False}})
    reloaded = PeftModel.from_pretrained(reload_base, adapter_dir)
    del reloaded

    print("Smoke test PASSED: pipeline runs end to end with no exceptions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["qlora", "lora"], default="qlora")
    args = parser.parse_args()
    run_smoke_test(args.method)
