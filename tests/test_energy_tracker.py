import csv

from src.energy_tracker import sum_emissions_for_run


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