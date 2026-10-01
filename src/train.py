"""Unified training entrypoint -- the fine-tuning method (full_ft / lora /
qlora) is selected entirely by the config YAML passed via --config, so this
script has no per-method branching. Also drives the intermediate-checkpoint
evaluations (every `eval_every_pct` of training) that feed the Green Gap
curve.
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must be set before the CUDA caching allocator is first used (any earlier
# ad-hoc/generate process does the same -- see scripts/measure_baseline.py).
# Doesn't lower the underlying memory requirement, but removes fragmentation
# OOMs on top of it. run_matrix.sh -> run_experiment.py -> this module, so a
# single setdefault here covers the real matrix entrypoint too.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from src.data import load_gsm8k, tokenize_dataset
from src.energy_tracker import EnergyRun, sum_emissions_for_run, summarize_sessions
from src.evaluate import EVAL_BATCH_SIZE, MID_TRAIN_EVAL_BATCH_SIZE, evaluate_model
from src.models import load_model, load_tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class GreenGapCheckpointCallback(TrainerCallback):
    """Evaluates on a small eval subsample every `eval_every_pct` of training
    and pairs the resulting accuracy with cumulative energy (from the GPU
    power poller) so we can plot accuracy vs. cumulative energy mid-run."""

    def __init__(
        self, model, tokenizer, eval_ds, energy_run, eval_every_pct,
        output_path, n_eval_subsample=100, eval_batch_size=MID_TRAIN_EVAL_BATCH_SIZE,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_ds = eval_ds.select(range(min(n_eval_subsample, len(eval_ds))))
        self.energy_run = energy_run
        self.eval_every_pct = eval_every_pct
        self.output_path = output_path
        self.eval_batch_size = eval_batch_size
        self.records = self._load_records()
        last_progress = max((float(record["progress"]) for record in self.records), default=0.0)
        self.next_threshold = last_progress + eval_every_pct

    def _load_records(self):
        if not os.path.exists(self.output_path):
            return []
        with open(self.output_path, newline="") as f:
            return [
                {
                    "step": int(row["step"]),
                    "progress": float(row["progress"]),
                    "accuracy": float(row["accuracy"]),
                    "cumulative_energy_kwh": float(row["cumulative_energy_kwh"]),
                }
                for row in csv.DictReader(f)
            ]

    def _append_record(self, record):
        write_header = not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        self.records.append(record)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 and torch.cuda.is_available():
            print(
                f"[MEM] step={state.global_step} "
                f"allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB "
                f"reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB "
                f"peak={torch.cuda.max_memory_allocated() / 1e9:.3f} GB"
            )
        if state.max_steps <= 0:
            return control
        progress = state.global_step / state.max_steps
        if progress >= self.next_threshold:
            if torch.cuda.is_available():
                print(
                    f"[MEM BEFORE EVAL] step={state.global_step} "
                    f"allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB "
                    f"reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB"
                )

            # eval_batch_size is deliberately smaller than the end-of-run
            # eval's: this runs with the optimizer state for the next steps
            # still resident, so the eval has to fit alongside it. Progress
            # lines are off here because the surrounding [MEM BEFORE/AFTER
            # EVAL] lines already mark the eval window in the log.
            accuracy, _ = evaluate_model(
                self.model, self.tokenizer, self.eval_ds,
                batch_size=self.eval_batch_size, log_every_n_batches=0,
            )

            if torch.cuda.is_available():
                print(
                    f"[MEM AFTER EVAL] step={state.global_step} "
                    f"allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB "
                    f"reserved={torch.cuda.memory_reserved() / 1e9:.3f} GB"
                )

            cumulative_kwh = (
                self.energy_run.poller.energy_kwh_elapsed() if self.energy_run.poller else None
            )
            self._append_record({
                "step": state.global_step,
                "progress": progress,
                "accuracy": accuracy,
                "cumulative_energy_kwh": cumulative_kwh,
            })
            self.next_threshold += self.eval_every_pct
        return control


def build_training_args(config, run_output_dir):
    return TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=config["seed"],
    )


def run(config_path, n_train_samples=None, n_eval_samples=None, seed_override=None, rank_override=None):
    config = load_config(config_path)

    if seed_override is not None:
        config["seed"] = seed_override
    if rank_override is not None and config.get("lora", {}).get("enabled"):
        config["lora"]["r"] = rank_override

    set_seed(config["seed"])

    lora_cfg = config.get("lora", {}) or {}
    run_id = f"{config['method']}"
    if lora_cfg.get("enabled"):
        run_id += f"_r{lora_cfg['r']}"
    run_id += f"_seed{config['seed']}"

    run_output_dir = os.path.join(config["output_dir"], run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    tokenizer = load_tokenizer(config["model_id"])
    model = load_model(config)
    if torch.cuda.is_available():
        print(f"Loaded model CUDA memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

    train_ds = load_gsm8k(tokenizer, "train", n_samples=n_train_samples)
    eval_ds = load_gsm8k(tokenizer, "test", n_samples=n_eval_samples)
    tokenized_train = tokenize_dataset(train_ds, tokenizer, config.get("max_seq_length", 512))

    training_args = build_training_args(config, run_output_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_train, data_collator=data_collator
    )

    # One line per EnergyRun session, so a crash-and-resume reports wall-clock
    # and peak power for the whole run rather than only the last session.
    energy_sessions_path = os.path.join(run_output_dir, "energy_sessions.jsonl")
    energy = EnergyRun(project_name=f"green-gap-{run_id}", session_log_path=energy_sessions_path)
    checkpoint_curve_path = os.path.join(run_output_dir, "checkpoint_curve.csv")
    with energy:
        callback = GreenGapCheckpointCallback(
            model, tokenizer, eval_ds, energy, config.get("eval_every_pct", 0.2),
            checkpoint_curve_path,
        )
        trainer.add_callback(callback)
        last_checkpoint = get_last_checkpoint(run_output_dir)

        if last_checkpoint:
            print(f"Resuming {run_id} from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print(f"Starting fresh run: {run_id}")
            trainer.train()

    # Standalone eval: training is done, so this takes the full batch size
    # (the same one scripts/measure_baseline.py measured the zero-shot
    # baseline at) over all 1,319 test examples.
    accuracy, records = evaluate_model(model, tokenizer, eval_ds, batch_size=EVAL_BATCH_SIZE)
    energy_summary = summarize_sessions(energy_sessions_path) or energy.summary()
    emissions_summary = sum_emissions_for_run(
        f"green-gap-{run_id}", energy.output_dir
    )

    result = {
        "run_id": run_id,
        "method": config["method"],
        "rank": lora_cfg.get("r") if lora_cfg.get("enabled") else None,
        "seed": config["seed"],
        "accuracy": accuracy,
        "energy_kwh": emissions_summary["energy_consumed"] if emissions_summary else None,
        "co2e_kg": emissions_summary["emissions"] if emissions_summary else None,
        "wall_clock_s": energy_summary["wall_clock_s"],
        "peak_gpu_watts": energy_summary["peak_gpu_watts"],
        # Max of earlier sessions (training) and this process (incl. the final eval).
        "peak_gpu_memory_bytes": max(
            (
                v for v in (
                    energy_summary.get("peak_gpu_memory_bytes"),
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                )
                if v is not None
            ),
            default=None,
        ),
    }

    with open(os.path.join(run_output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(run_output_dir, "eval_records.json"), "w") as f:
        json.dump(records, f, indent=2)

    model.save_pretrained(os.path.join(run_output_dir, "final_adapter_or_model"))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Green Gap training entrypoint")
    parser.add_argument("--config", required=True, help="Path to a config YAML (configs/*.yaml)")
    parser.add_argument("--n_train_samples", type=int, default=None)
    parser.add_argument("--n_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    parser.add_argument("--rank", type=int, default=None, help="Override the LoRA rank (r) for ablation runs")
    args = parser.parse_args()

    result = run(
        args.config,
        n_train_samples=args.n_train_samples,
        n_eval_samples=args.n_eval_samples,
        seed_override=args.seed,
        rank_override=args.rank,
    )
    print(json.dumps(result, indent=2))
