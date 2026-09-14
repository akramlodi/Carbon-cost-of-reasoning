## Context

`energy_fix.md`'s fixes are implemented and unit-tested (synthetic CSV aggregation passes). Two things remain before the real experiment matrix can run: (1) GPU-dependent verification that couldn't be done in the sandboxed environment, and (2) a gap in `run_matrix.sh` that would make it crash or misallocate cost if run as-is.

Target hardware for everything below: AWS `g5.xlarge` (1x A10G, 24GB) — already provisioned. Nothing here needs the L40S.

## Part 1: `run_matrix.sh` has no per-method GPU routing

The script loops `full_ft`, `lora`, `qlora` back-to-back for every seed, all under one invocation with a single `GG_GPU_MODEL` value. But the three methods need different hardware per the project's compute plan: `full_ft` needs `g6e.xlarge` (L40S, 48GB) and will OOM on `g5.xlarge`'s 24GB; `lora`/`qlora` are meant to run on the cheaper `g5.xlarge` and don't need the L40S's cost.

As written, there's no way to run "just the g5-appropriate methods" on `g5.xlarge` and "just full_ft" separately on `g6e.xlarge` without hand-editing the script each time.

**Add a `GG_METHODS` filter** (comma-separated, e.g. `GG_METHODS=lora,qlora`) that restricts which method loops actually execute, defaulting to all three if unset (so existing single-instance usage isn't broken). Concretely: skip the `run configs/full_ft.yaml ...` calls (both core-seed loop and any ablations) when `full_ft` isn't in the requested set, and same for `lora`/`qlora`. The existing per-run `result.json`-exists skip logic should stay untouched — this is a separate, additive filter on top of it.

## Part 2: Two GPU-dependent checks, as ready-to-run scripts

Provide these as standalone scripts (or clearly copy-pasteable shell commands) rather than descriptions — they'll be run directly on the provisioned `g5.xlarge`, inside `tmux`, the same way the earlier batching tests were.

### Check A: interrupt-and-resume energy accounting

Goal: prove that when a run crashes and resumes, the final `energy_kwh`/`co2e_kg` in `result.json` reflect the **sum** of both tracker sessions, not just the post-resume session — this is the actual bug `sum_emissions_for_run` was written to fix, and it's never been exercised against a real interrupted `Trainer.train()`.

Suggested approach: use the cheapest real condition available (QLoRA, low rank, `--n_train_samples` set small enough to guarantee multiple epochs/checkpoints within a minute or two) so the check is fast and cheap. Start the run, let it get partway through (enough to pass at least one `save_strategy` checkpoint boundary), interrupt it (Ctrl+C or `kill`), then rerun the identical command and confirm it resumes via `get_last_checkpoint` rather than starting over.

After it completes: read the raw rows in `results/emissions/emissions.csv` for that run's `project_name`, manually sum `energy_consumed` and `emissions` by hand (or with a one-off script), and compare that manual sum against what actually landed in the run's `result.json`. They must match exactly. Print both values side by side so this comparison is visible in the output, not just asserted.

### Check B: checkpoint-curve reload-on-resume

Goal: the synthetic unit test proved `sum_emissions_for_run`'s math is correct, but never exercised whether `GreenGapCheckpointCallback` actually reloads and continues appending to `checkpoint_curve.csv` after a real interrupt, rather than overwriting it or losing pre-crash rows.

Using the same interrupted run from Check A (or a fresh short one): after the crash-and-resume completes, inspect `checkpoint_curve.csv` for that run and confirm (a) it contains records from both before and after the interruption — i.e., `step` values aren't all clustered only after the resume point, and (b) there are no duplicate `step` entries from the pre-crash session being re-recorded on top of themselves after resume.

### Check C (optional — not required given sequential g5-then-g6e execution, but cheap if you want the extra confidence)

Run two short experiments concurrently against the same `emissions.csv` (e.g. two QLoRA runs with different seeds, launched a few seconds apart in separate terminals/tmux panes), and confirm each run's final `result.json` reflects only its own `project_name`'s rows, not a value bled in from the other concurrently-running experiment. Skip this if time-constrained — the current plan runs everything sequentially, so this failure mode won't actually occur in practice.

## What to report back

For each check: pass/fail, and for Check A specifically, the actual manually-summed value vs. what landed in `result.json` — don't just say "matched," show the two numbers. If anything fails, stop there rather than proceeding to the real matrix — these are the last gate before spending real money on 13 runs.