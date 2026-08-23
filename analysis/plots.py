"""Accuracy-vs-energy curves and Green Gap visualization, built from
results/metrics.csv (one row per finished run) and any results/runs/**/
checkpoint_curve.csv files (one row per intermediate checkpoint eval, written
by src.train's GreenGapCheckpointCallback).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import green_gap_curve

METHOD_COLORS = {"full_ft": "#d62728", "lora": "#1f77b4", "qlora": "#2ca02c"}


def load_metrics(path="results/metrics.csv"):
    return pd.read_csv(path)


def load_checkpoint_curves(runs_dir="results"):
    frames = []
    for path in glob.glob(os.path.join(runs_dir, "**", "checkpoint_curve.csv"), recursive=True):
        df = pd.read_csv(path)
        df["run_id"] = os.path.basename(os.path.dirname(path))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_accuracy_vs_energy(metrics_df, out_path="analysis/accuracy_vs_energy.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in metrics_df.groupby("method"):
        ax.scatter(
            group["energy_kwh"], group["accuracy"],
            label=method, color=METHOD_COLORS.get(method), s=60,
        )
    ax.set_xlabel("Energy consumed (kWh)")
    ax.set_ylabel("GSM8K exact-match accuracy")
    ax.set_title("Accuracy vs. Energy by Fine-Tuning Method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_green_gap_curve(checkpoint_df, out_path="analysis/green_gap_curve.png", baseline_accuracy=None):
    if baseline_accuracy is not None:
        checkpoint_df = green_gap_curve(checkpoint_df, baseline_accuracy)

    fig, ax = plt.subplots(figsize=(7, 5))
    for run_id, group in checkpoint_df.groupby("run_id"):
        group = group.sort_values("cumulative_energy_kwh")
        ax.plot(group["cumulative_energy_kwh"], group["accuracy"], marker="o", label=run_id)
        if "green_gap_point" in group.columns:
            gap_points = group[group["green_gap_point"]]
            if not gap_points.empty:
                ax.scatter(
                    gap_points["cumulative_energy_kwh"], gap_points["accuracy"],
                    color="black", marker="x", s=80, zorder=5,
                )
    ax.set_xlabel("Cumulative energy consumed (kWh)")
    ax.set_ylabel("GSM8K exact-match accuracy")
    ax.set_title("The Green Gap: Accuracy vs. Cumulative Energy")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_rei_bar(metrics_df, out_path="analysis/rei_by_method.png"):
    summary = metrics_df.groupby("method")["rei_kwh"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary.index, summary.values, color=[METHOD_COLORS.get(m, "#888888") for m in summary.index])
    ax.set_ylabel("REI (delta-accuracy / kWh)")
    ax.set_title("Reasoning Efficiency Index by Method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Green Gap analysis plots")
    parser.add_argument("--metrics_csv", default="results/metrics.csv")
    parser.add_argument("--out_dir", default="analysis")
    parser.add_argument(
        "--baseline_accuracy", type=float, default=None,
        help="Zero-shot base-model accuracy; if given, flags the Green Gap point on the checkpoint curve",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_df = load_metrics(args.metrics_csv)

    if metrics_df.empty:
        print(f"No rows in {args.metrics_csv} yet -- run experiments before plotting.")
        return

    print(f"Wrote {plot_accuracy_vs_energy(metrics_df, os.path.join(args.out_dir, 'accuracy_vs_energy.png'))}")
    print(f"Wrote {plot_rei_bar(metrics_df, os.path.join(args.out_dir, 'rei_by_method.png'))}")

    checkpoint_df = load_checkpoint_curves("results")
    if checkpoint_df.empty:
        print("No checkpoint_curve.csv files found -- skipping Green Gap curve plot.")
        return

    out_path = os.path.join(args.out_dir, "green_gap_curve.png")
    print(f"Wrote {plot_green_gap_curve(checkpoint_df, out_path, args.baseline_accuracy)}")


if __name__ == "__main__":
    main()
