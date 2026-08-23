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
and needs fixing in configs/full_ft.yaml (use the module-prefix breakdown
below it to find the real name). The final trainable/total ratio should
land close to the ~1.87B / 5.1B (~37%) ballpark from CHANGES.md -- far from
that means the patterns are systematically too broad or too narrow.

Per-pattern element counts can double-count when two patterns match the
same tensor (e.g. a per-layer-embedding table whose name contains both
"embed_tokens" and "per_layer") -- that's a reporting artifact of scanning
each pattern independently, not a bug in the real freeze logic, which
freezes each parameter once regardless of how many patterns match it. The
"actual total frozen" figure reported at the end is the ground truth.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import load_model, load_tokenizer
from src.train import load_config

# Scanned across ALL parameter names regardless of where they fall in
# iteration order (a plain first-N dump can easily land entirely inside one
# submodule, as it did for vision_tower on the first run of this script).
KEYWORDS_OF_INTEREST = [
    "vision", "audio", "draft", "speculat", "embed", "per_layer", "language_model",
]


def main():
    config = load_config("configs/full_ft.yaml")
    frozen_patterns = config.get("frozen_modules", [])
    if not frozen_patterns:
        print("configs/full_ft.yaml has no frozen_modules set -- nothing to check.")
        return

    print(f"Loading {config['model_id']} (bf16, no quantization, no LoRA)...")
    load_tokenizer(config["model_id"])  # not used below, just mirrors the real load path
    model = load_model(config)  # applies the freeze itself; prints its own trainable-params line

    all_names_and_sizes = [(n, p.numel()) for n, p in model.named_parameters()]
    total_params = sum(numel for _, numel in all_names_and_sizes)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    actual_frozen = total_params - trainable_params

    print("\n--- configs/full_ft.yaml pattern match report ---")
    unmatched_patterns = []
    for pattern in frozen_patterns:
        matches = [(n, numel) for n, numel in all_names_and_sizes if pattern in n]
        total = sum(numel for _, numel in matches)
        print(f"{pattern!r}: {len(matches)} params matched, {total:,} elements (may overlap with other patterns)")
        if not matches:
            unmatched_patterns.append(pattern)

    if unmatched_patterns:
        print(f"\nWARNING: these frozen_modules patterns matched NOTHING: {unmatched_patterns}")
        print("See the full-checkpoint keyword scan below to find the real name, or confirm the component is genuinely absent.")
    else:
        print("\nAll configured patterns matched at least one parameter.")

    print(f"\nActual total frozen (ground truth, no double-counting): {actual_frozen:,} / {total_params:,}")

    print("\n--- Full-checkpoint keyword scan (case-insensitive, all parameter names) ---")
    for keyword in KEYWORDS_OF_INTEREST:
        matches = [(n, numel) for n, numel in all_names_and_sizes if keyword in n.lower()]
        total = sum(numel for _, numel in matches)
        example = matches[0][0] if matches else "(none found anywhere in the checkpoint)"
        print(f"{keyword!r}: {len(matches)} params, {total:,} elements -- e.g. {example}")

    print("\n--- Unique module-name prefixes up to depth 3 (e.g. model.vision_tower) ---")
    prefixes = sorted({".".join(n.split(".")[:3]) for n, _ in all_names_and_sizes})
    for prefix in prefixes:
        print(prefix)


if __name__ == "__main__":
    main()
