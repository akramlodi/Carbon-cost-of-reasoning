import csv

from src.energy_tracker import append_session, sum_emissions_for_run, summarize_sessions


def write_emissions_csv(path, rows):
    fieldnames = ["project_name", "energy_consumed", "emissions"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_sum_emissions_for_run_filters_and_sums_sessions(tmp_path):
    write_emissions_csv(
        tmp_path / "emissions.csv",
        [
            {"project_name": "green-gap-qlora_seed1", "energy_consumed": "0.25", "emissions": "0.10"},
            {"project_name": "green-gap-lora_seed1", "energy_consumed": "9.0", "emissions": "8.0"},
            {"project_name": "green-gap-qlora_seed1", "energy_consumed": "0.75", "emissions": "0.30"},
        ],
    )

    assert sum_emissions_for_run("green-gap-qlora_seed1", str(tmp_path)) == {
        "energy_consumed": 1.0,
        "emissions": 0.4,
    }
    assert sum_emissions_for_run("missing", str(tmp_path)) is None


def test_summarize_sessions_sums_wall_clock_and_takes_peak_across_resume(tmp_path):
    path = str(tmp_path / "energy_sessions.jsonl")
    append_session(path, {"wall_clock_s": 8750.0, "peak_gpu_watts": 190.0, "peak_gpu_memory_bytes": 14_000})
    append_session(path, {"wall_clock_s": 2.0, "peak_gpu_watts": 58.0, "peak_gpu_memory_bytes": 10_000})

    assert summarize_sessions(path) == {
        "wall_clock_s": 8752.0,
        "peak_gpu_watts": 190.0,
        "peak_gpu_memory_bytes": 14_000,
        "n_sessions": 2,
    }


def test_summarize_sessions_missing_file_and_no_poller(tmp_path):
    assert summarize_sessions(str(tmp_path / "missing.jsonl")) is None

    path = str(tmp_path / "energy_sessions.jsonl")
    append_session(path, {"wall_clock_s": 5.0, "peak_gpu_watts": None})
    assert summarize_sessions(path)["peak_gpu_watts"] is None
