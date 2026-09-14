## Context

Two related bugs in energy/emissions accounting were found while reviewing `src/energy_tracker.py` and `src/train.py` ahead of running the real (billed) experiment matrix on AWS. Both affect `energy_kwh` and `co2e_kg` in `result.json` / `results/metrics.csv` — the core measurements the paper's REI and Green Gap analysis are built on — so these need to be fixed before any billed run, not after.

## Bug 1: `read_latest_emissions_row` doesn't filter by run

`results/emissions/emissions.csv` is a single file shared across the entire experiment matrix — every run's `EmissionsTracker` (one per `EnergyRun` in `train.py`) appends a row to the same file, keyed by `project_name` (which is set to `run_id`). `read_latest_emissions_row()` in `src/energy_tracker.py` ignores this entirely and just returns the positionally-last row in the file:

```python
with open(path) as f:
    rows = list(csv.DictReader(f))
return rows[-1] if rows else None
```

This only happens to work if every run executes strictly sequentially, start-to-finish, with no overlap. If two runs are ever executed concurrently (e.g. one on `g5.xlarge`, one on `g6e.xlarge`, to cut wall-clock time — this was discussed as an option), CSV rows from different runs can interleave, and this function can silently return an unrelated run's energy number into a completely different run's `result.json`.

## Bug 2: crash-and-resume undercounts both `energy_kwh` and `co2e_kg`

`scripts/run_matrix.sh` skips already-completed runs by checking for `result.json`, and `Trainer` resumes from its own checkpoint via `get_last_checkpoint` in `src/train.py` — but the `EnergyRun` context manager in `train.py` starts a **fresh** `EmissionsTracker` session every time `run()` is called, with no awareness of prior sessions for the same `run_id`. If a run crashes and is resumed:

- The pre-crash `EmissionsTracker` session already wrote its own row to `emissions.csv` (via CodeCarbon's normal per-session behavior).
- The resumed run starts a new session, which only tracks energy/emissions from the resume point forward.
- `result.json` ends up populated from either `tracker.stop()`'s return value (used for `co2e_kg`) or `read_latest_emissions_row()` (used for `energy_kwh`) — both of which reflect **only the most recent session**, not the sum across all sessions for that `run_id`.

Net effect: any run that needed a resume will have its pre-crash energy and emissions silently missing from both `energy_kwh` and `co2e_kg`, with no error or warning. This directly corrupts REI for that run.

## Required fix

Replace `read_latest_emissions_row` in `src/energy_tracker.py` with a function that sums across every session belonging to a given `run_id`, rather than reading the positionally-last row:

```python
def sum_emissions_for_run(run_id, output_dir="results/emissions", csv_name="emissions.csv"):
    """Sums energy_consumed and emissions across every tracker session
    (project_name == run_id) in the shared emissions.csv. Correctly handles
    both crash-and-resume (multiple sessions, same run_id, need to be summed)
    and concurrent runs (interleaved rows from different run_ids, need to be
    filtered) -- unlike reading the positionally-last row in the file.
    """
    path = os.path.join(output_dir, csv_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("project_name") == run_id]
    if not rows:
        return None
    return {
        "energy_consumed": sum(float(r["energy_consumed"]) for r in rows),
        "emissions": sum(float(r["emissions"]) for r in rows),
    }
```

**Before wiring this in, verify the actual column names against the installed CodeCarbon version's `emissions.csv` header** — `energy_consumed`, `emissions`, and `project_name` are the standard CodeCarbon field names, but this should be confirmed against the real file (or CodeCarbon's changelog for the pinned version) rather than assumed, since schemas have drifted across versions before.

Then update `src/train.py`'s `run()` function to use this for **both** `energy_kwh` and `co2e_kg`, instead of the current mix of `read_latest_emissions_row()` (for energy) and `energy.summary()["emissions_kg"]` / `tracker.stop()`'s return value (for CO2e) — both numbers should come from the same summed-and-filtered source:

```python
emissions_summary = sum_emissions_for_run(run_id, energy.output_dir)
result = {
    ...
    "energy_kwh": emissions_summary["energy_consumed"] if emissions_summary else None,
    "co2e_kg": emissions_summary["emissions"] if emissions_summary else None,
    ...
}
```

## Also fix: Green Gap checkpoint records lost on a mid-training crash

Separately, in `train.py`: `GreenGapCheckpointCallback.records` accumulates in memory and is only written to `checkpoint_curve.csv` **after** `trainer.train()` returns successfully:

```python
with energy:
    ...
    trainer.train(...)

if callback.records:
    pd.DataFrame(callback.records).to_csv(...)
```

If `trainer.train()` raises partway through, every intermediate accuracy/cumulative-energy checkpoint collected up to that point is lost, even though the underlying model training can resume fine from the Trainer's own checkpoint. Apply the same append-as-you-go pattern already used in `scripts/measure_baseline.py` (which writes each record to disk immediately rather than accumulating and writing once at the end): have `GreenGapCheckpointCallback` append each record to a CSV/JSONL file as it's produced, not just at the end of training. On resume, it should also be able to pick up any records already on disk from a prior session for the same `run_id`, rather than starting the curve over.

## Lower priority — document rather than necessarily fix now

`wall_clock_s`, `peak_gpu_watts`, and `mean_gpu_watts` in `EnergyRun.summary()` have the same "only reflects the current session" limitation, since `GPUPowerPoller.samples` resets on every fresh `EnergyRun` instantiation. These don't feed into REI directly, so this doesn't need to block the billed runs — but if any run actually ends up needing a resume, add a note to `CHANGES.md` for that run stating that its reported wall-clock time and peak/mean wattage reflect only the post-resume session, not the full run, so this doesn't quietly skew any later secondary analysis or plot that uses those fields.

## Testing before trusting this on a billed run

1. **Unit-level**: with synthetic `emissions.csv` rows (multiple `project_name` values, some runs with 2+ rows simulating a resume), confirm `sum_emissions_for_run` returns the correct filtered-and-summed values for each `run_id`, and `None` for a `run_id` with no matching rows.
2. **Integration**: on the cheapest real condition (QLoRA, low rank, few steps — e.g. a short manual run with `--n_train_samples` set small), deliberately interrupt training partway through (Ctrl+C), confirm `get_last_checkpoint` resumes correctly, and confirm the final `result.json`'s `energy_kwh`/`co2e_kg` reflect the sum of both sessions rather than just the post-resume session — check this by also inspecting the raw `emissions.csv` rows for that `run_id` and manually summing them to compare.
3. **Concurrency check** (if you intend to run `g5.xlarge` and `g6e.xlarge` in parallel to save wall-clock time): run two short experiments concurrently against the same `emissions.csv`, confirm each run's `result.json` reflects only its own rows, not a value from the other concurrently-running experiment.

Only proceed to the full 13-run matrix after these pass.