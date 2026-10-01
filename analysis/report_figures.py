"""Static figures for README.md / EXPERIMENT_REPORT.md, built from the
g5.xlarge LoRA/QLoRA matrix in results-g5-backup/ and the zero-shot baseline.

Usage:
    python analysis/report_figures.py [--results results-g5-backup] [--out docs/figures]
"""
import argparse
import csv
import glob
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette roles (validated two-slot categorical set + text/surface tokens).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
LORA = "#2a78d6"
QLORA = "#eb6834"
BASELINE = "#0b0b0b"
FIXED = "#2a78d6"   # diverging pole: wrong -> right
BROKEN = "#e34948"  # diverging pole: right -> wrong
METHOD_COLOR = {"lora": LORA, "qlora": QLORA}
METHOD_LABEL = {"lora": "LoRA", "qlora": "QLoRA"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": TEXT,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_2,
})


def load(results_dir, baseline_path):
    with open(os.path.join(results_dir, "metrics.csv"), newline="") as f:
        runs = list(csv.DictReader(f))
    for r in runs:
        for k in ("accuracy", "energy_kwh", "co2e_kg", "wall_clock_s", "baseline_accuracy"):
            r[k] = float(r[k])
        with open(os.path.join(results_dir, "runs", r["method"], r["run_id"], "result.json")) as f:
            r["result"] = json.load(f)
        with open(os.path.join(results_dir, "runs", r["method"], r["run_id"], "eval_records.json")) as f:
            r["records"] = json.load(f)
    with open(baseline_path) as f:
        baseline = [json.loads(line) for line in f]
    return runs, baseline


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run_label(r):
    return f"{METHOD_LABEL[r['method']]} r={r['rank']} · seed {r['seed']}"


def ordered(runs):
    return sorted(runs, key=lambda r: (r["method"], int(r["rank"]) != 16, int(r["rank"]), int(r["seed"])))


def fig_accuracy(runs, baseline_acc, out_dir):
    rows = ordered(runs)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r["accuracy"] * 100, height=0.62, color=METHOD_COLOR[r["method"]])
        ax.text(r["accuracy"] * 100 - 1, yi, f"{r['accuracy'] * 100:.1f}%", va="center", ha="right",
                color="white", fontsize=9.5, fontweight="bold")
    ax.axvline(baseline_acc * 100, color=BASELINE, lw=2, ls="--")
    ax.text(baseline_acc * 100 + 0.8, max(y) + 0.6, f"Zero-shot base model\n{baseline_acc * 100:.1f}%",
            color=TEXT, fontsize=10, va="top", fontweight="bold")
    ax.set_yticks(y, [run_label(r) for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("GSM8K test accuracy (%), 1,319 questions")
    ax.grid(axis="y", visible=False)
    ax.set_title("Every fine-tuned run scored below the model it started from")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LORA, QLORA)]
    ax.legend(handles, ["LoRA", "QLoRA"], loc="lower left", bbox_to_anchor=(0.68, 0.0))
    save(fig, out_dir, "fig1_accuracy_vs_baseline.png")


def fig_energy_scatter(runs, baseline_acc, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter([0], [baseline_acc * 100], s=140, marker="*", color=BASELINE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.annotate("Zero-shot base model\n0 kWh of fine-tuning", (0, baseline_acc * 100), xytext=(0.03, 86.5),
                color=TEXT, fontsize=10, fontweight="bold")
    for method in ("lora", "qlora"):
        pts = [r for r in runs if r["method"] == method]
        ax.scatter([r["energy_kwh"] for r in pts], [r["accuracy"] * 100 for r in pts], s=70,
                   color=METHOD_COLOR[method], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=METHOD_LABEL[method])
    outlier = next(r for r in runs if r["run_id"] == "lora_r16_seed1")
    ax.annotate("LoRA seed 1: energy inflated\nby unbatched mid-training evals",
                (outlier["energy_kwh"], outlier["accuracy"] * 100), xytext=(0.255, 70.5),
                color=TEXT_2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate("", xy=(0.29, 64), xytext=(0.012, baseline_acc * 100 - 1.2),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5, ls=(0, (4, 3))))
    ax.text(0.075, 72.5, "~0.3 kWh spent, 23–28 points lost", color=TEXT_2, fontsize=10.5, rotation=-26)
    ax.set_xlim(-0.02, 0.36)
    ax.set_ylim(55, 93)
    ax.set_xlabel("Training energy per run (kWh, CodeCarbon: GPU + CPU + RAM)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("More energy, lower accuracy: every REI is negative")
    ax.legend(loc="lower left")
    save(fig, out_dir, "fig2_accuracy_vs_energy.png")


def fig_green_gap(results_dir, baseline, out_dir):
    subset_acc = sum(r["correct"] for r in baseline[:100])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in ("lora", "qlora"):
        by_step = {}
        for path in glob.glob(os.path.join(results_dir, "runs", method, "*", "checkpoint_curve.csv")):
            if "seed999" in path:
                continue
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    by_step.setdefault(int(row["step"]), []).append(
                        (float(row["cumulative_energy_kwh"]), float(row["accuracy"]) * 100))
        steps = sorted(by_step)
        x = [0] + [statistics.mean(e for e, _ in by_step[s]) for s in steps]
        mean = [subset_acc] + [statistics.mean(a for _, a in by_step[s]) for s in steps]
        lo = [subset_acc] + [min(a for _, a in by_step[s]) for s in steps]
        hi = [subset_acc] + [max(a for _, a in by_step[s]) for s in steps]
        color = METHOD_COLOR[method]
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(x, mean, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
                label=f"{METHOD_LABEL[method]} (mean of 5 runs, band = min–max)")
        ax.text(x[-1] + 0.006, mean[-1], METHOD_LABEL[method], color=TEXT, va="center", fontweight="bold")
    ax.axhline(subset_acc, color=BASELINE, lw=2, ls="--")
    ax.text(0.005, subset_acc + 0.8, f"Base model on the same 100 questions: {subset_acc}%", color=TEXT,
            fontsize=10, fontweight="bold")
    ax.annotate("Drop happens in the first\n20% of training (~0.6 epoch)", xy=(0.058, 55.5), xytext=(0.085, 47.5),
                color=TEXT_2, fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlim(-0.005, 0.33)
    ax.set_ylim(45, 100)
    ax.set_xlabel("Cumulative GPU energy during training (kWh, NVML poller)")
    ax.set_ylabel("Accuracy on 100-question checkpoint subset (%)")
    ax.set_title("Green Gap curve: accuracy never recovers as energy accumulates")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.83), fontsize=9.5)
    save(fig, out_dir, "fig3_green_gap_curve.png")


def fig_flips(runs, baseline, out_dir):
    rows = ordered(runs)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        broken = sum(b["correct"] and not f["correct"] for b, f in zip(baseline, r["records"]))
        fixed = sum(f["correct"] and not b["correct"] for b, f in zip(baseline, r["records"]))
        ax.barh(yi, -broken, height=0.62, color=BROKEN)
        ax.barh(yi, fixed, height=0.62, color=FIXED)
        ax.text(-broken - 8, yi, str(broken), va="center", ha="right", color=TEXT, fontsize=9.5)
        ax.text(fixed + 8, yi, str(fixed), va="center", ha="left", color=TEXT, fontsize=9.5)
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(y, [run_label(r) for r in rows])
    ax.set_xlim(-470, 130)
    ax.set_xticks([-400, -300, -200, -100, 0, 100], ["400", "300", "200", "100", "0", "100"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Number of test questions (of 1,319)")
    ax.text(-460, max(y) + 0.9, "◀ Base model right → fine-tuned wrong", color=TEXT, fontsize=10, fontweight="bold")
    ax.text(5, max(y) + 0.9, "Wrong → right ▶", color=TEXT, fontsize=10, fontweight="bold")
    ax.set_ylim(-0.7, max(y) + 1.5)
    ax.set_title("Fine-tuning broke ~9 questions for every 1 it fixed", pad=14)
    save(fig, out_dir, "fig4_question_flips.png")


def fig_lora_vs_qlora(runs, out_dir):
    # LoRA seed 1 is excluded: its energy/time include unbatched mid-training evals.
    core = {m: [r for r in runs if r["method"] == m and r["rank"] == "16" and r["run_id"] != "lora_r16_seed1"]
            for m in ("lora", "qlora")}
    panels = [
        ("Accuracy (%)", lambda r: r["accuracy"] * 100, "{:.1f}%"),
        ("Training energy (kWh)", lambda r: r["energy_kwh"], "{:.3f}"),
        ("Training time (h)", lambda r: r["wall_clock_s"] / 3600, "{:.2f} h"),
        ("Peak GPU memory (GB)", lambda r: r["result"]["peak_gpu_memory_bytes"] / 1e9, "{:.1f} GB"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2))
    for ax, (title, fn, fmt) in zip(axes, panels):
        vals = [statistics.mean(fn(r) for r in core[m]) for m in ("lora", "qlora")]
        ax.bar([0, 1], vals, width=0.62, color=[LORA, QLORA])
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, fmt.format(v), ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1], ["LoRA", "QLoRA"])
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("QLoRA vs LoRA (r=16) on a 24 GB A10G: 25% less memory, but 16% more energy and 3.4 points lower",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.01, -0.03, "LoRA: seeds 2–3 (seed 1 excluded, see report §6.4). QLoRA: seeds 1–3.",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, out_dir, "fig5_lora_vs_qlora.png")


BASE_EXAMPLE = (
    "Here is the step-by-step calculation:\n\n"
    "1. Total eggs laid per day: 16\n"
    "2. Eggs eaten for breakfast: 3\n"
    "3. Eggs used for muffins: 4\n"
    "4. Total used = 3 + 4 = 7 eggs\n"
    "5. Remaining = 16 − 7 = 9 eggs\n"
    "6. Money made = 9 × \\$2 = \\$18\n\n"
    "#### 18"
)
TARGET_EXAMPLE = (
    "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck\n"
    "eggs a day.\n"
    "She makes 9 * 2 = \\$<<9*2=18>>18 every day\n"
    "at the farmer's market.\n"
    "#### 18"
)


def fig_style_mismatch(baseline, out_dir):
    base_chars = statistics.mean(len(r["generated_text"]) for r in baseline)
    fig = plt.figure(figsize=(11, 5.6))
    fig.suptitle("Why accuracy dropped: we trained the model to imitate a terser style than its own",
                 x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=TEXT)
    fig.text(0.02, 0.885, "Same test question (Janet's ducks). Left: what the base model writes. "
             "Right: the GSM8K reference solution style used as the training target.",
             color=TEXT_2, fontsize=10)
    for x, title, body, color in (
        (0.02, "Base model's own answer (abridged)", BASE_EXAMPLE, LORA),
        (0.51, "GSM8K training target", TARGET_EXAMPLE, QLORA),
    ):
        ax = fig.add_axes([x, 0.30, 0.47, 0.54])
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white",
                                   edgecolor=GRID, lw=1.2))
        ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, transform=ax.transAxes, facecolor=color, lw=0))
        ax.text(0.03, 0.95, title, transform=ax.transAxes, va="center", color="white", fontweight="bold")
        ax.text(0.03, 0.84, body, transform=ax.transAxes, va="top", family="DejaVu Sans Mono",
                fontsize=9.5, color=TEXT, linespacing=1.45)
    ax = fig.add_axes([0.2, 0.06, 0.62, 0.16])
    bars = [("Base model output", base_chars, LORA), ("GSM8K target", 290, QLORA)]
    for i, (label, v, c) in enumerate(bars):
        ax.barh(1 - i, v, height=0.6, color=c)
        ax.text(v + 15, 1 - i, f"{v:,.0f} chars on average", va="center", color=TEXT, fontsize=10)
    ax.set_yticks([1, 0], [b[0] for b in bars])
    ax.set_xlim(0, 1400)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=9)
    fig.text(0.02, -0.04, "99% of GSM8K targets contain <<a op b = c>> calculator markup, written for a "
             "calculator tool this model does not have.", color=MUTED, fontsize=9)
    save(fig, out_dir, "fig6_style_mismatch.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results-g5-backup")
    parser.add_argument("--baseline", default="baseline_records.jsonl")
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs, baseline = load(args.results, args.baseline)
    baseline_acc = runs[0]["baseline_accuracy"]
    fig_accuracy(runs, baseline_acc, args.out)
    fig_energy_scatter(runs, baseline_acc, args.out)
    fig_green_gap(args.results, baseline, args.out)
    fig_flips(runs, baseline, args.out)
    fig_lora_vs_qlora(runs, args.out)
    fig_style_mismatch(baseline, args.out)


if __name__ == "__main__":
    main()
