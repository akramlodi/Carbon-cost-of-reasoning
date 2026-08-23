"""Full experiment runner for a single condition: trains via src.train.run,
computes REI against a supplied zero-shot baseline, and appends one row to
results/metrics.csv -- the aggregated table the experiment matrix builds up.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import compute_rei, compute_rei_co2
from src.train import run as run_training

METRICS_CSV = "results/metrics.csv"
METRICS_FIELDS = [
    "run_id", "method", "rank", "seed", "gpu_model", "region",
    "wall_clock_s", "energy_kwh", "co2e_kg", "accuracy",
    "baseline_accuracy", "delta_accuracy", "rei_kwh", "rei_co2",
]


def append_metrics_row(row, path=METRICS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run a single Green Gap experiment condition")
    parser.add_argument("--config", required=True, help="Path to a config YAML (configs/*.yaml)")
    parser.add_argument(
        "--baseline_accuracy", type=float, required=True,
        help="Zero-shot base-model accuracy, used as the REI reference point",
    )
    parser.add_argument("--gpu_model", default=os.environ.get("GG_GPU_MODEL", "unknown"))
    parser.add_argument("--region", default=os.environ.get("GG_REGION", "unknown"))
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    parser.add_argument("--rank", type=int, default=None, help="Override the LoRA rank for ablation runs")
    parser.add_argument("--n_train_samples", type=int, default=None)
    parser.add_argument("--n_eval_samples", type=int, default=None)
    args = parser.parse_args()

    result = run_training(
        args.config,
        n_train_samples=args.n_train_samples,
        n_eval_samples=args.n_eval_samples,
        seed_override=args.seed,
        rank_override=args.rank,
    )

    delta_accuracy = result["accuracy"] - args.baseline_accuracy
    row = {
        "run_id": result["run_id"],
        "method": result["method"],
        "rank": result["rank"],
        "seed": result["seed"],
        "gpu_model": args.gpu_model,
        "region": args.region,
        "wall_clock_s": result["wall_clock_s"],
        "energy_kwh": result["energy_kwh"],
        "co2e_kg": result["co2e_kg"],
        "accuracy": result["accuracy"],
        "baseline_accuracy": args.baseline_accuracy,
        "delta_accuracy": delta_accuracy,
        "rei_kwh": compute_rei(delta_accuracy, result["energy_kwh"]),
        "rei_co2": compute_rei_co2(delta_accuracy, result["co2e_kg"]),
    }
    append_metrics_row(row)
    print(f"Logged {row['run_id']} to {METRICS_CSV}")


if __name__ == "__main__":
    main()
