# The Green Gap: A Sustainability Analysis of Fine-Tuning Methods for Reasoning Models

## Overview

This project measures the environmental cost — energy consumption (kWh) and CO2 emissions — of three fine-tuning methodologies applied to a small reasoning-capable language model, and weighs that cost against the accuracy gained on a multi-step mathematical reasoning benchmark.

**Base model:** `google/gemma-4-E2B-it` (instruction-tuned; 2.3B effective parameters, Apache 2.0)
**Benchmark:** GSM8K (grade-school math word problems requiring chain-of-thought reasoning)
**Methods compared:** Full Fine-Tuning, LoRA, QLoRA

**Core contributions:**
1. **The Green Gap** — the point in training where marginal accuracy gained per additional kWh drops sharply, i.e. where environmental cost stops buying meaningful reasoning improvement.
2. **The Reasoning Efficiency Index (REI)** — a standardized metric for comparing fine-tuning methods on environmental efficiency rather than raw accuracy alone.

```
REI = ΔAccuracy (over zero-shot base model) / Energy consumed (kWh)
```

(A CO2e-denominated variant may also be reported for cross-region comparability, since kWh-to-CO2e conversion depends on local grid carbon intensity.)

---

## Repository Structure

```
green-gap/
├── README.md
├── requirements.txt
├── configs/
│   ├── full_ft.yaml
│   ├── lora.yaml
│   └── qlora.yaml
├── src/
│   ├── data.py              # GSM8K loading, prompt formatting, tokenization
│   ├── train.py             # unified training entrypoint (method selected via config)
│   ├── models.py            # model/tokenizer loading, LoRA/QLoRA adapter setup
│   ├── energy_tracker.py    # CodeCarbon/Zeus wrapper, GPU power polling
│   ├── evaluate.py          # GSM8K exact-match scoring, answer extraction
│   └── metrics.py           # REI calculation, Green Gap curve generation
├── scripts/
│   ├── smoke_test.py        # tiny end-to-end run for Colab free-tier T4
│   ├── run_experiment.py    # full experiment runner (single condition)
│   └── run_matrix.sh        # loops through the full experiment matrix
├── notebooks/
│   └── colab_smoke_test.ipynb
├── results/
│   ├── logs/                # per-run training logs
│   ├── emissions/           # CodeCarbon output CSVs
│   └── metrics.csv          # aggregated accuracy/energy/CO2/REI per run
└── analysis/
    └── plots.py             # accuracy-vs-energy curves, Green Gap visualization
```

---

## Environment Setup

### Phase 1 — Free Colab (T4, 16GB): pipeline validation only

Use this phase to confirm the code runs correctly. Do **not** treat any numbers produced here as real results — shared/virtualized GPU time on free Colab is not reliable for energy measurement, and T4 cannot run full fine-tuning at all.

```python
!pip install -q transformers peft bitsandbytes accelerate datasets codecarbon trl
```

Runs: LoRA ✅, QLoRA ✅, Full FT ❌ (will OOM — expected, not a bug)

### Phase 2 — Rented GPU: measured experiments

Provision a single, fixed-spec instance per condition (RunPod, Lambda Labs, or similar):
- Full FT → A100 40GB
- LoRA → RTX 4090 (24GB) or L4 (24GB)
- QLoRA → L4 (16–24GB)

Keep the exact GPU model, driver version, and CUDA version identical across repeated runs of the same condition — this is a controlled variable in the study and should be reported in the paper.

```bash
pip install -r requirements.txt
```

---

## `requirements.txt` (draft)

```
transformers>=4.45
peft>=0.13
bitsandbytes>=0.44
accelerate>=1.0
datasets>=3.0
trl>=0.11
codecarbon>=2.5
torch>=2.4
scikit-learn
pandas
matplotlib
pyyaml
```

---

## Data

**Dataset:** `openai/gsm8k` (`main` config) via Hugging Face `datasets`
- Train split: ~7,473 examples
- Test split: ~1,319 examples
- Format: question + step-by-step solution ending in `#### <final_numeric_answer>`

**Prompt template** (chain-of-thought, consistent across all three methods): the instruction text below is sent as a single user turn through the tokenizer's chat template (`google/gemma-4-E2B-it` is instruction-tuned and expects its own turn-formatted input, not a raw completion-style string):
```
Question: {question}
Answer: Let's think step by step.
```
Target: an assistant turn containing the reference solution text, ending in the `####` answer marker (see `src/data.py::format_example`, which builds both the eval-time prompt and the training-time full text via `tokenizer.apply_chat_template`).

---

## Methodology

### Shared training config (held constant across all three methods)
- Same train/test split, same random seed(s) — recommend 3 seeds per condition
- Same number of epochs (start with 3; confirm via smoke test that this is a reasonable budget)
- Same effective batch size (use gradient accumulation to match across memory-constrained vs full FT setups)
- Same evaluation protocol and prompt template
- Mixed precision: bf16 throughout

### Condition A — Full Fine-Tuning
- Only the language-model transformer backbone (attention + MLP layers, ~1.88B params) is trainable. `google/gemma-4-E2B-it` loads as ~5.1B raw parameters, not the "2.3B effective" marketing figure; the gap is the embedder (incl. Per-Layer Embeddings), audio encoder, and vision encoder, all of which are **frozen** for this condition (see `configs/full_ft.yaml`'s `frozen_modules`, verified against the real checkpoint via `scripts/check_full_ft_scope.py` — this checkpoint has no separate speculative-decoding drafter module).
  - Google freezes the audio/vision encoders during gemma-4's own pretraining, and GSM8K is text-only, so there's no gradient signal for them regardless.
  - LoRA/QLoRA (Conditions B/C) only adapt backbone attention/MLP projections and never touch the embedder — training the embedder here too would conflate "fine-tuning method" with "training scope" and undermine the cross-method comparison.
  - Reasoning ability lives in the backbone; GSM8K introduces no new vocabulary.
  - Keeps optimizer-state memory within an A100 40GB budget instead of needing an 80GB card for all 5.1B raw params.
- AdamW optimizer, fp32 master weights (or bf16 with loss scaling if memory-constrained)
- Gradient checkpointing enabled to control memory

### Condition B — LoRA
- Base model frozen, loaded in bf16
- Adapter target modules: attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) at minimum; consider adding MLP projections
- Primary rank: r=16, alpha=32, dropout=0.05
- Secondary ranks for the Green Gap ablation curve: r=8, r=32

### Condition C — QLoRA
- Base model loaded in 4-bit NF4 quantization (`bitsandbytes`)
- Same adapter config as LoRA condition B for direct comparability
- Same rank ablation (r=8, r=16, r=32)

---

## Energy & Carbon Tracking

Wrap every training run (not just the final experiment matrix — the smoke test too, to validate the tracker itself works) with `codecarbon.EmissionsTracker`:

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="green-gap", output_dir="results/emissions")
tracker.start()
# ... training loop ...
emissions_kg = tracker.stop()
```

Log per run:
- Total energy consumed (kWh) — GPU + CPU + RAM as reported by CodeCarbon
- Estimated CO2e (kg), using CodeCarbon's regional grid intensity data — **record the region of the rented instance**, since this materially affects CO2e even for identical kWh
- Wall-clock training time
- Peak GPU memory (`torch.cuda.max_memory_allocated()`)

Consider cross-checking CodeCarbon's GPU energy numbers against a second tool (e.g. `zeus` from the ML Energy Initiative) on at least one run per condition, since shared-infrastructure energy attribution has known noise — useful for a methods/limitations note in the paper.

---

## Evaluation

- Metric: exact-match accuracy on the final numeric answer (parse both prediction and reference for the `####` marker; compare numerically, not as strings, to avoid formatting mismatches like `12` vs `12.0`)
- Evaluate on the full GSM8K test set (1,319 examples) after training completes
- Also log accuracy at intermediate checkpoints (e.g. every 20% of training) to construct the Green Gap curve (accuracy vs. cumulative energy)

---

## Experiment Matrix

| Condition | Rank | Reps | GPU | Priority |
|---|---|---|---|---|
| Full FT | — | 3 | A100 40GB | Core |
| LoRA | r=16 | 3 | RTX 4090 / L4 | Core |
| QLoRA | r=16 | 3 | L4 | Core |
| LoRA | r=8 | 1 | RTX 4090 / L4 | Ablation |
| LoRA | r=32 | 1 | RTX 4090 / L4 | Ablation |
| QLoRA | r=8 | 1 | L4 | Ablation |
| QLoRA | r=32 | 1 | L4 | Ablation |

**Total core runs:** 9 (for statistical variance on the headline comparison)
**Total ablation runs:** 4 (to plot the Green Gap curve across capacity/rank)

For every run, record: method, rank (if applicable), seed, GPU model, region, wall-clock time, energy (kWh), CO2e (kg), final accuracy, and REI.

---

## Smoke Test — Run This First on Free Colab (T4)

**Goal:** confirm the entire pipeline executes without errors, end to end, before spending any money. This is not a real experiment — it uses a tiny data subset and 1 epoch purely to catch bugs (data formatting issues, OOM on the LoRA/QLoRA configs, tokenizer mismatches, CodeCarbon initialization failures, checkpoint saving).

**What this smoke test validates:**
- [ ] Gemma 4 E2B loads correctly in 4-bit (QLoRA) and bf16 (LoRA) on a T4
- [ ] GSM8K loads, formats, and tokenizes without errors
- [ ] LoRA adapter attaches to the correct target modules
- [ ] Training loop runs for a handful of steps without OOM or NaN loss
- [ ] CodeCarbon tracker starts, logs, and stops cleanly, producing a CSV
- [ ] Evaluation script correctly extracts and compares numeric answers
- [ ] Checkpoints/adapters save and reload correctly

**What this smoke test does NOT validate:** real accuracy numbers, real energy numbers, or full fine-tuning (T4 cannot run it — expect and ignore the OOM if you try).

The actual implementation lives in `scripts/smoke_test.py`, not inline here — it's picked up several fixes since this section was first drafted (chat-template prompting, dynamic per-batch padding, a `peft`-version-driven `target_modules` fix, T4 memory-fragmentation mitigations) that would just go stale if duplicated into this file as a second copy. Run it with:
```bash
python scripts/smoke_test.py --method qlora
python scripts/smoke_test.py --method lora
```

**Expected runtime on free T4:** a few minutes. If this fails, fix it before spending a cent on rented GPU time — every bug caught here saves real money later.

**Next step after a clean smoke test pass:** repeat the same script structure with the Full FT config (no quantization, no LoRA, all params trainable) on a rented A100, since that path cannot be validated on T4.

---

## Reproducibility Notes

- Fix and record random seeds for data shuffling, weight initialization, and any sampling in generation
- Report exact library versions (`pip freeze > requirements_lock.txt` after final runs)
- Report exact GPU model, region, and cloud provider for every run (needed for CO2e reporting)
- Report Gemma 4 E2B checkpoint/revision hash used

---

## Open Questions to Resolve Before Full Matrix Runs

- [ ] Confirm 3 epochs is an appropriate training budget (check via smoke-test-scale loss curves, or a slightly larger pilot run)
- [ ] Decide whether QLoRA and LoRA should target identical modules for strict comparability, or method-optimal configs
- [ ] Decide on the CO2e grid-intensity assumption to use if the rented instance's region isn't directly supported by CodeCarbon's database
- [ ] Decide statistical test for comparing REI across methods (e.g. paired t-test across seeds)

---

## License

Code: TBD (recommend MIT or Apache 2.0 to match Gemma's license)
Model: Gemma 4 E2B is distributed under Apache 2.0 by Google DeepMind — review the Gemma Terms of Use for any additional prohibited-use restrictions before publishing derived checkpoints.