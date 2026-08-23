"""Diagnostic-only check for the Full FT frozen_modules scope in
configs/full_ft.yaml.

Loads google/gemma-4-E2B in bf16 (no quantization, no LoRA -- the exact Full
FT load path), applies the freeze via the real src.models.load_model() code
path, and reports which frozen_modules patterns actually matched real
parameter names and how many elements each froze. Runs NO training steps,
so it's safe on a free T4 even though a real Full FT training run isn't
(see README) -- this only needs to load the model once.

The frozen_modules list in configs/full_ft.yaml is currently a best-guess
based on common HF naming conventions, not confirmed against the actual
checkpoint (see CHANGES.md, "Change 1"). Run this once before spending
rented-A100 time on a real Full FT run:

    python scripts/check_full_ft_scope.py

Read the "per-pattern match report": any pattern with 0 matches is wrong
and needs fixing in configs/full_ft.yaml (use the printed name dump to find
the real prefix). The final trainable/total ratio should land close to the
~1.87B / 5.1B (~37%) ballpark from CHANGES.md -- far from that means the
patterns are systematically too broad or too narrow.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_model, load_tokenizer
from src.train import load_config


def main():
    config = load_config("configs/full_ft.yaml")
    frozen_patterns = config.get("frozen_modules", [])
    if not frozen_patterns:
        print("configs/full_ft.yaml has no frozen_modules set -- nothing to check.")
        return

    print(f"Loading {config['model_id']} (bf16, no quantization, no LoRA)...")
    load_tokenizer(config["model_id"])  # not used below, just mirrors the real load path
    model = load_model(config)  # applies the freeze itself; prints its own trainable-params line

    print("\n--- Per-pattern match report ---")
    unmatched_patterns = []
    for pattern in frozen_patterns:
        matches = [(n, p.numel()) for n, p in model.named_parameters() if pattern in n]
        total = sum(numel for _, numel in matches)
        print(f"{pattern!r}: {len(matches)} params matched, {total:,} elements frozen")
        if not matches:
            unmatched_patterns.append(pattern)

    if unmatched_patterns:
        print(f"\nWARNING: these frozen_modules patterns matched NOTHING: {unmatched_patterns}")
        print("Either they're wrong (check the name dump below) or that component isn't present under this name.")
    else:
        print("\nAll patterns matched at least one parameter.")

    print("\n--- All top-level module names (first token of each dotted param name) ---")
    top_level = sorted({name.split(".")[0] for name, _ in model.named_parameters()})
    for name in top_level:
        print(name)

    print("\n--- First 80 full parameter names (for manual inspection) ---")
    for name, _ in list(model.named_parameters())[:80]:
        print(name)


if __name__ == "__main__":
    main()
