"""Verify interrupted QLoRA resume energy accounting on a real GPU.

Run from the repository root inside tmux:
    python scripts/verify_gpu_resume.py

The script starts a short run, waits until its first checkpoint-curve record
exists, sends SIGINT, then reruns the identical command. It compares the raw
CodeCarbon rows with result.json and checks that checkpoint_curve.csv contains
pre- and post-interruption records without duplicate steps.
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def wait_for_checkpoint_and_curve(curve_path, run_dir, process, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        checkpoints = list(run_dir.glob("checkpoint-*"))
        if curve_path.exists() and len(read_rows(curve_path)) >= 1 and checkpoints:
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"training exited before the first curve record (exit={process.returncode})"
            )
        time.sleep(2)
    process.send_signal(signal.SIGINT)
    process.wait()
    raise RuntimeError(
        f"timed out waiting for a curve row and Trainer checkpoint in {run_dir}; "
        "training was stopped"
    )


def interrupt_after_first_checkpoint(command, curve_path, run_dir, timeout_s):
    print("Starting the intentional pre-resume run:", " ".join(command), flush=True)
    process = subprocess.Popen(command)
    try:
        wait_for_checkpoint_and_curve(curve_path, run_dir, process, timeout_s)
        print(
            f"Curve record and Trainer checkpoint observed in {run_dir}; sending SIGINT",
            flush=True,
        )
        process.send_signal(signal.SIGINT)
        process.wait()
    except Exception:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait()
        raise
    if process.returncode == 0:
        raise RuntimeError(
            "the short run completed before interruption; increase --n-train-samples "
            "or --epochs so the test exercises resume"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--n-train-samples", type=int, default=64)
    parser.add_argument("--n-eval-samples", type=int, default=20)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    run_id = f"qlora_r16_seed{args.seed}"
    run_dir = repo_root / "results" / "runs" / "qlora" / run_id
    curve_path = run_dir / "checkpoint_curve.csv"
    result_path = run_dir / "result.json"
    emissions_path = repo_root / "results" / "emissions" / "emissions.csv"

    if result_path.exists():
        raise RuntimeError(f"{result_path} already exists; choose another --seed")
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; choose another --seed or remove it")

    command = [
        args.python, "-m", "src.train",
        "--config", args.config,
        "--seed", str(args.seed),
        "--n_train_samples", str(args.n_train_samples),
        "--n_eval_samples", str(args.n_eval_samples),
    ]
    interrupt_after_first_checkpoint(command, curve_path, run_dir, args.timeout_s)
    pre_resume_rows = read_rows(curve_path)
    if not pre_resume_rows:
        raise RuntimeError("no pre-resume checkpoint curve rows were persisted")

    print("Rerunning the identical command; Trainer should resume from its checkpoint", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"resumed training failed with exit={completed.returncode}")

    if not result_path.exists():
        raise RuntimeError(f"resumed run did not produce {result_path}")
    if not emissions_path.exists():
        raise RuntimeError(f"missing CodeCarbon output {emissions_path}")

    project_name = f"green-gap-{run_id}"
    matching_rows = [
        row for row in read_rows(emissions_path)
        if row.get("project_name") == project_name
    ]
    if len(matching_rows) < 2:
        raise RuntimeError(
            f"expected at least two CodeCarbon sessions for {project_name}, "
            f"found {len(matching_rows)}"
        )
    manual_energy = sum(float(row["energy_consumed"]) for row in matching_rows)
    manual_emissions = sum(float(row["emissions"]) for row in matching_rows)
    with result_path.open() as stream:
        result = json.load(stream)

    print("Check A: interrupt-and-resume energy accounting")
    print(f"  raw-row sum energy_kwh : {manual_energy:.12g}")
    print(f"  result.json energy_kwh : {result['energy_kwh']:.12g}")
    print(f"  raw-row sum co2e_kg    : {manual_emissions:.12g}")
    print(f"  result.json co2e_kg    : {result['co2e_kg']:.12g}")
    if result["energy_kwh"] != manual_energy or result["co2e_kg"] != manual_emissions:
        raise RuntimeError("Check A FAILED: result.json does not match summed CodeCarbon rows")
    print("  PASS")

    final_rows = read_rows(curve_path)
    steps = [int(row["step"]) for row in final_rows]
    pre_steps = [int(row["step"]) for row in pre_resume_rows]
    post_steps = [step for step in steps if step > max(pre_steps)]
    print("Check B: checkpoint-curve reload-on-resume")
    print(f"  pre-resume steps : {pre_steps}")
    print(f"  final steps      : {steps}")
    if len(final_rows) <= len(pre_resume_rows) or not post_steps:
        raise RuntimeError("Check B FAILED: no post-resume checkpoint records found")
    if len(steps) != len(set(steps)):
        raise RuntimeError("Check B FAILED: duplicate checkpoint steps found")
    print("  PASS")


if __name__ == "__main__":
    main()