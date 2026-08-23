"""REI (Reasoning Efficiency Index) calculation and Green Gap curve generation.

REI = delta_accuracy (over zero-shot base model) / energy consumed (kWh)
A CO2e-denominated variant is also provided for cross-region comparability.
"""
import pandas as pd


def compute_rei(delta_accuracy, energy_kwh):
    if energy_kwh in (None, 0) or pd.isna(energy_kwh):
        return None
    return delta_accuracy / energy_kwh


def compute_rei_co2(delta_accuracy, co2e_kg):
    if co2e_kg in (None, 0) or pd.isna(co2e_kg):
        return None
    return delta_accuracy / co2e_kg


def add_rei_columns(df, baseline_accuracy):
    """Given a metrics.csv-shaped dataframe (one row per run), add
    delta_accuracy / rei_kwh / rei_co2 columns relative to baseline_accuracy."""
    df = df.copy()
    df["delta_accuracy"] = df["accuracy"] - baseline_accuracy
    df["rei_kwh"] = df.apply(
        lambda r: compute_rei(r["delta_accuracy"], r.get("energy_kwh")), axis=1
    )
    df["rei_co2"] = df.apply(
        lambda r: compute_rei_co2(r["delta_accuracy"], r.get("co2e_kg")), axis=1
    )
    return df


def green_gap_curve(checkpoint_df, baseline_accuracy, marginal_rei_floor_pct=0.1):
    """checkpoint_df: rows of (run_id, cumulative_energy_kwh, accuracy), one
    row per intermediate checkpoint eval, sorted by training progress.

    Returns the same rows with delta_accuracy, marginal_rei (marginal
    delta-accuracy per marginal kWh between consecutive checkpoints), and a
    green_gap_point flag marking the first checkpoint per run where
    marginal_rei drops below `marginal_rei_floor_pct` of that run's running
    max -- a first-pass heuristic for "the Green Gap": the point where
    additional energy stops buying meaningful reasoning improvement.
    """
    df = checkpoint_df.sort_values(["run_id", "cumulative_energy_kwh"]).copy()
    df["delta_accuracy"] = df["accuracy"] - baseline_accuracy

    out = []
    for _, group in df.groupby("run_id"):
        group = group.sort_values("cumulative_energy_kwh").reset_index(drop=True)
        d_energy = group["cumulative_energy_kwh"].diff()
        d_acc = group["delta_accuracy"].diff()
        group["marginal_rei"] = d_acc / d_energy
        running_max = group["marginal_rei"].cummax()
        group["green_gap_point"] = group["marginal_rei"] < marginal_rei_floor_pct * running_max
        out.append(group)
    return pd.concat(out, ignore_index=True)


def aggregate_metrics_csv(records, path="results/metrics.csv"):
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    return df
