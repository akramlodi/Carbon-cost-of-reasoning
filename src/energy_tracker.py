"""CodeCarbon wrapper plus a lightweight NVML-based GPU power poller.

CodeCarbon gives an authoritative total (kWh, kg CO2e) once a run finishes,
but the Green Gap curve needs *cumulative* energy at intermediate checkpoints
mid-run. The poller integrates instantaneous GPU power draw over time to
approximate that running total; it's a secondary signal, not a replacement
for CodeCarbon's final numbers -- see README's note on cross-checking with a
second tool (e.g. `zeus`).
"""
import csv
import os
import threading
import time

from codecarbon import EmissionsTracker

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class GPUPowerPoller:
    """Polls instantaneous GPU power draw (W) on a background thread."""

    def __init__(self, interval_s=1.0, device_index=0):
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def _poll_loop(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        while not self._stop_event.is_set():
            watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            self.samples.append((time.time(), watts))
            time.sleep(self.interval_s)

    def start(self):
        if not _NVML_AVAILABLE:
            return
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_s * 2)

    def peak_watts(self):
        return max((w for _, w in self.samples), default=0.0)

    def mean_watts(self):
        if not self.samples:
            return 0.0
        return sum(w for _, w in self.samples) / len(self.samples)

    def energy_kwh_elapsed(self):
        """Trapezoidal integration of power samples collected so far."""
        if len(self.samples) < 2:
            return 0.0
        energy_ws = sum(
            0.5 * (w0 + w1) * (t1 - t0)
            for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:])
        )
        return energy_ws / 3600.0 / 1000.0


class EnergyRun:
    """Context manager wrapping CodeCarbon + optional GPU power polling for one run."""

    def __init__(self, project_name, output_dir="results/emissions", poll_gpu=True):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.tracker = EmissionsTracker(project_name=project_name, output_dir=output_dir)
        self.poller = GPUPowerPoller() if poll_gpu else None
        self.emissions_kg = None
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.tracker.start()
        if self.poller:
            self.poller.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emissions_kg = self.tracker.stop()
        if self.poller:
            self.poller.stop()
        self.end_time = time.time()
        return False

    @property
    def wall_clock_s(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def summary(self):
        return {
            "emissions_kg": self.emissions_kg,
            "wall_clock_s": self.wall_clock_s,
            "peak_gpu_watts": self.poller.peak_watts() if self.poller else None,
            "mean_gpu_watts": self.poller.mean_watts() if self.poller else None,
        }


def read_latest_emissions_row(output_dir="results/emissions", csv_name="emissions.csv"):
    """CodeCarbon appends one row per run to output_dir/emissions.csv; this
    pulls the most recent row so callers can read fields like energy_consumed
    (kWh) that aren't returned directly by `tracker.stop()`.
    """
    path = os.path.join(output_dir, csv_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None
