# The Green Gap: Experiment Report (LoRA / QLoRA on AWS g5.xlarge)

This report gives the full details behind the summary in [README.md](README.md): what we set out to measure, how the pipeline was built and checked, exactly what ran on AWS, what came out, and why. The raw outputs are in `results-g5-backup/`.

---

## 1. What we set out to measure

The project asks whether the energy and carbon spent fine-tuning a small reasoning model actually buy better reasoning, and how that trade-off differs between fine-tuning methods.

Two quantities were defined up front:

- **Reasoning Efficiency Index (REI):** accuracy gained over the zero-shot base model, per unit of energy.
  ```
  REI_kWh = (accuracy_finetuned − accuracy_zero_shot) / energy_kWh
  REI_CO2 = (accuracy_finetuned − accuracy_zero_shot) / CO2e_kg
  ```
  Accuracy is a fraction (0–1), so an REI of −0.87 /kWh means each kWh moved accuracy down by 0.87 accuracy units (87 points) per kWh. In practice a run uses about 0.3 kWh, so the total change is about 26 points.
- **The Green Gap:** the point during training where extra kWh stop buying meaningful accuracy. It is read from accuracy-vs-cumulative-energy curves recorded at 20% intervals through training.

The expected outcome was a positive accuracy gain for every method, with LoRA/QLoRA giving most of Full Fine-Tuning's gain at a fraction of its energy.

---

## 2. Setup

### 2.1 Model, data, prompt

| Item | Value |
|---|---|
| Model | `google/gemma-4-E2B-it` (instruction-tuned; ~5.1B raw params, "2.3B effective") |
| Benchmark | GSM8K `main` (`openai/gsm8k`): 7,473 train / 1,319 test |
| Prompt | One user turn through the tokenizer's chat template: `Question: {question}\nAnswer: Let's think step by step. End your response with only: #### <number>` |
| Training target | Assistant turn holding the GSM8K reference solution (`answer` field), ending in `#### <number>` |
| Scoring | Numeric exact match on the last number after `####` (falls back to the last number in the text), tolerance 1e-4 |
| Decoding | Greedy, `max_new_tokens=768`, left-padded batches of 16, contiguous slices of the test set |

The baseline and every fine-tuned eval use the same code path: `src/data.py::format_example` builds the prompt, `src/evaluate.py::generate_batch` generates, and `extract_final_answer` scores. The same 1,319 questions are scored in the same order, which we checked by matching question text record-for-record.

### 2.2 Hardware and software

| Item | Value |
|---|---|
| Instance | AWS **g5.xlarge**, `us-east-1` (N. Virginia) |
| GPU | 1× NVIDIA A10G, 24 GB (~22 GB usable) |
| CPU / RAM | 4 vCPU AMD EPYC 7R32 / 16 GB |
| OS | Amazon Linux 2023 (kernel 6.12) |
| Python / CodeCarbon | 3.12.14 / 3.3.1 |
| Key libraries | `transformers`, `peft>=0.19` (needed to wrap Gemma-4's `Gemma4ClippableLinear`), `bitsandbytes`, `torchao>=0.16`, `pynvml`; see `requirements.txt` |

### 2.3 Training configuration

Both conditions share everything except how the base model is held in memory.

| Setting | LoRA (`configs/lora.yaml`) | QLoRA (`configs/qlora.yaml`) |
|---|---|---|
| Base weights | bf16, frozen | 4-bit NF4 + double quantization, bf16 compute, frozen |
| Adapter | r=16, α=32, dropout 0.05 | same |
| Target modules | PEFT ≥0.19 Gemma-4 defaults (LM attention + MLP projections; vision/audio skipped) | same |
| Epochs | 3 (1,404 optimizer steps) | same |
| Batch | 2 per device × 8 accumulation = 16 effective | same |
| Learning rate | 2e-4 (Trainer default linear decay, AdamW) | same |
| Max sequence length | 512 (only 7/7,473 examples exceed it) | same |
| Precision | bf16, gradient checkpointing on | same |
| Mid-training eval | every 20% of training, first 100 test questions, batch 4 | same |
| Final eval | all 1,319 test questions, batch 16 | same |
| Rank ablations | r=8 and r=32 via `--rank`, seed 1 | same |

`configs/full_ft.yaml` exists, with LR 2e-5, batch 4×4, and only the 1.88B-param language backbone trainable. It is planned for a g6e.xlarge (L40S, 48 GB) and **has not been run**; the 24 GB A10G cannot hold it.

### 2.4 Energy and carbon measurement

- **CodeCarbon** (`EmissionsTracker`, machine mode) gives the authoritative per-run kWh (GPU + CPU + RAM) and CO2e. For `us-east-1` it applied **0.369 kg CO2e/kWh** with PUE 1.0. Each run's total is the sum of every CodeCarbon session tagged with that run's project name, so a crash-and-resume is counted in full.
- **NVML power poller** (1 s samples) gives cumulative GPU-only energy at each mid-training checkpoint for the Green Gap curve. It reads about 10% lower than CodeCarbon because it excludes CPU/RAM.
- **Measurement window:** the tracker wraps `trainer.train()`, so it **includes the five 100-question mid-training evals but excludes the final 1,319-question eval**, which took about 50–55 minutes per run at 0.40 examples/s. Reported energy is the cost of training, not of training plus final scoring.

---

## 3. How we got here: pipeline development and checks

Chronologically (full write-ups are in [CHANGES.md](CHANGES.md), [energy_fix.md](energy_fix.md) and [gpu verification.md](gpu%20verification.md)):

1. **Smoke tests on free Colab T4** (`scripts/smoke_test.py`): LoRA and QLoRA trained 10 steps with falling loss, saved and reloaded adapters, and CodeCarbon produced CSVs. Full FT was not tested; it is expected to run out of memory on a T4.
2. **Full FT scope (Change 1):** froze vision/audio towers, their fusion projections and the embedding tables, leaving 1,877,105,920 / 5,104,297,504 params (36.78%) trainable. Verified on both the base and `-it` checkpoints with `scripts/check_full_ft_scope.py`.
3. **Switch to `-it` + chat template (Change 3).** Also fixed a double-BOS bug (`add_special_tokens=False` wherever already-templated text is tokenized).
4. **Terse final-answer instruction (Change 4):** stops the model's essay-style hedging from running past the token budget and gives answer extraction a reliable `####` anchor.
5. **Energy accounting fixes (energy_fix.md):** per-run filtering and summing of CodeCarbon rows across resume sessions; checkpoint-curve rows appended to disk as they are produced so a crash doesn't lose them.
6. **Out-of-memory at step 0 on g5.xlarge (Change 5):** Gemma-4's 262,144-token vocabulary makes the logits tensor huge. Fixed with per-device batch 8→2 and accumulation 2→8 (effective batch unchanged at 16), gradient checkpointing on for LoRA (plus `enable_input_require_grads`), and `expandable_segments`. Passed a 200-sample memory gate (`lora_r16_seed999`, the extra CodeCarbon rows in `emissions.csv`).
7. **Batched evaluation (Change 6):** the one-at-a-time final eval took over an hour per run. It now runs in left-padded batches of 16 (final) and 4 (mid-training), shared with `measure_baseline.py` so the baseline and the fine-tuned evals cannot drift apart. Verified on GPU with `scripts/verify_batched_eval.py`: batched answers are deterministic run-to-run, and they differ from unbatched answers on a small, non-directional fraction of near-tie examples (expected numerical behaviour, analysed in CHANGES.md).

---

## 4. Zero-shot baseline

Measured with `scripts/measure_baseline.py` on the same g5.xlarge (`baseline_accuracy.json`, `baseline_records.jsonl`):

| Metric | Value |
|---|---|
| Accuracy | **88.78%** (1,171 / 1,319) |
| Generations that hit the token limit | 60 (4.5%) |
| Mean generated length | ~1,070 characters of step-by-step reasoning |
| Accuracy on the first 100 questions (the mid-training eval subset) | 93% |

**Discrepancy:** the baseline was generated with `max_new_tokens=786`, while every fine-tuned eval used the project standard of 768 (almost certainly a typo at launch). Only 61 baseline generations ran longer than 768 tokens, and 9 of those were scored correct. At 768 tokens the baseline would therefore be between **88.10% and 88.78%**. That moves every Δaccuracy below by at most 0.7 points and changes no conclusion. A re-measure at 768 is listed under future work.

---

## 5. The matrix that ran

```bash
GG_METHODS=lora,qlora GG_BASELINE_ACCURACY=0.8877937831690674 \
  GG_GPU_MODEL=A10G-24GB GG_REGION=us-east-1 ./scripts/run_matrix.sh
```

All runs ran one after another on one g5.xlarge from 2026-09-30 03:00 to 2026-10-01 11:00 UTC: 6 core runs (LoRA and QLoRA at r=16 × seeds 1, 2, 3) plus 4 rank ablations (r=8 and r=32 × LoRA and QLoRA, seed 1). Every run finished 3 epochs; training loss fell from about 3.3 to 0.79 (LoRA) and 0.84 (QLoRA), with no NaNs and no out-of-memory errors (peak allocated about 14.2 GB LoRA, 10.7 GB QLoRA).

---

## 6. Results

### 6.1 Per run

| Run | Accuracy | Δ vs baseline | Energy (kWh) | CO2e (kg) | Train time (h) | REI (/kWh) | Peak GPU W |
|---|---|---|---|---|---|---|---|
| lora_r16_seed1 † | 66.11% | −22.67 pt | 0.330 | 0.122 | 2.43 | −0.687 | n/a |
| lora_r16_seed2 | 65.73% | −23.05 pt | 0.266 | 0.098 | 1.77 | −0.868 | 193 |
| lora_r16_seed3 | 64.82% | −23.96 pt | 0.272 | 0.101 | 1.81 | −0.879 | 198 |
| lora_r8_seed1 | 64.52% | −24.26 pt | 0.275 | 0.102 | 1.87 | −0.882 | 194 |
| lora_r32_seed1 | 64.67% | −24.11 pt | 0.267 | 0.099 | 1.78 | −0.902 | 206 |
| qlora_r16_seed1 | 61.64% | −27.14 pt | 0.313 | 0.115 | 2.19 | −0.868 | 190 |
| qlora_r16_seed2 | 61.79% | −26.99 pt | 0.311 | 0.115 | 2.15 | −0.869 | 195 |
| qlora_r16_seed3 | 62.32% | −26.46 pt | 0.315 | 0.116 | 2.19 | −0.841 | 196 |
| qlora_r8_seed1 | 60.65% | −28.13 pt | 0.318 | 0.117 | 2.25 | −0.884 | 187 |
| qlora_r32_seed1 | 61.11% | −27.67 pt | 0.319 | 0.118 | 2.25 | −0.868 | 191 |

† See §6.4: this run's energy and time include slower, one-at-a-time mid-training evals, so they are not comparable with the others.

Total for the 10 runs: **2.99 kWh, 1.10 kg CO2e, 20.7 h** of training. Including the memory-gate runs it is 3.06 kWh. Final evals add about 9 h on top, outside the tracked window.

### 6.2 Per condition (r=16, 3 seeds, mean ± sd)

| Condition | Accuracy | Δ vs baseline | Energy (kWh) | CO2e (kg) | REI (/kWh) |
|---|---|---|---|---|---|
| Zero-shot baseline | 88.78% | — | 0 | 0 | — |
| LoRA r=16 | 65.55 ± 0.66% | −23.22 ± 0.66 pt | 0.289 ± 0.035 (0.269 excl. seed 1) | 0.107 ± 0.013 | −0.81 ± 0.11 (−0.87 excl. seed 1) |
| QLoRA r=16 | 61.92 ± 0.36% | −26.86 ± 0.36 pt | 0.313 ± 0.002 | 0.115 ± 0.001 | −0.86 ± 0.02 |

### 6.3 Green Gap curves (100-question subset, mean of 5 runs per method; baseline 93% on this subset)

| Training progress | LoRA acc | LoRA cumulative GPU kWh | QLoRA acc | QLoRA cumulative GPU kWh |
|---|---|---|---|---|
| 20% (step 281) | 61.0% | 0.053 | 56.8% | 0.059 |
| 40% (step 562) | 66.8% | 0.106 | 61.8% | 0.117 |
| 60% (step 843) | 63.4% | 0.157 | 63.0% | 0.175 |
| 80% (step 1124) | 63.8% | 0.208 | 59.2% | 0.233 |
| 100% (step 1404) | 64.8% | 0.260 | 62.2% | 0.290 |

On a 100-question sample, one question is one point, so the step-to-step wiggle (±5 points between runs) is mostly noise. The signal is the level: **every checkpoint of every run is 26–37 points below the base model's 93% on the same questions, from the first checkpoint onward.**

### 6.4 Data corrections made

- **`lora_r16_seed1` wall-clock, peak power and peak memory.** This was the first matrix run. Its final eval was still running one example at a time when it was interrupted. The eval was then batched (CHANGES.md Change 6) and the run resumed from `checkpoint-1404`, did zero further training steps, and ran the batched eval. The old code reported `wall_clock_s` (2.24 s), `peak_gpu_watts` (58.7 W) and `peak_gpu_memory_bytes` (10.8 GB) from the 0.5 s resume session only. We corrected the backup to `wall_clock_s = 8749.5` (sum of both CodeCarbon sessions) and set peak power and peak memory to null, because they cannot be recovered. The original values are kept under `correction` in its `result.json`. Energy and CO2e were already correct, since they sum both sessions. **Code fix:** `EnergyRun` now appends each session's wall-clock, peak power and peak memory to `energy_sessions.jsonl` in the run directory, and `result.json` aggregates across sessions (sum and max). Unit-tested in `tests/test_energy_tracker.py`.
- **`lora_r16_seed1` energy is not comparable.** Its training session ran the five mid-training evals one example at a time (before Change 6), which added about 40 minutes and about 0.06 kWh compared with seeds 2 and 3. Its accuracy is comparable, because its final eval used the same batched path. For energy comparisons, use seeds 2 and 3 (0.269 kWh mean), or rerun seed 1.
- **Baseline token budget** (786 vs 768): bounded in §4 and not corrected (needs a GPU).

---

## 7. Analysis: why fine-tuning made the model worse

### 7.1 It is not an evaluation artefact

- Same prompt construction, chat template, decoding settings, batch size, batch boundaries and answer extraction for the baseline and every fine-tuned eval (shared code, §2.1).
- Same 1,319 questions in the same order (verified).
- Results are tight across seeds (sd 0.4–0.7 points) and across ranks, so it is not a one-off bad run.
- Training itself was healthy: the loss converged smoothly and both methods reached the same loss range.

### 7.2 The training targets teach a weaker way of reasoning

The base model already solves GSM8K well **in its own style**: long, self-checking, step-by-step explanations (about 1,070 characters on average). The GSM8K reference solutions we trained on look like this:

```
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

| | Base model's own answers | GSM8K training targets |
|---|---|---|
| Mean length | ~1,070 chars | ~290 chars (sample of 100) |
| Style | Explains, checks, re-derives | Terse, one line per step |
| `<<a op b = c>>` calculator markup | none | 99% of solutions |

Supervised fine-tuning does what it is told: it makes the model reproduce this format. That has two costs:

1. **Less reasoning per answer.** The model learns to write about a quarter as much reasoning, which removes the intermediate checking that made the base model accurate.
2. **Calculator markup without a calculator.** The original GSM8K annotations were written for a tool that computes `<<48/2=24>>` during generation. Our model must produce both sides of each equation itself, so the markup only adds tokens where arithmetic errors can happen.

The low training loss (~0.8) confirms the model learned to imitate the targets. The imitation itself is what hurts accuracy.

### 7.3 The evidence

- **It happens immediately.** By the first checkpoint (20% of training, about 0.6 epochs) accuracy on the 100-question subset is already 57–61%, against 93% for the base model, and it never recovers. This is not slow overfitting over 3 epochs. It is a fast change of output style, which is what imitating short targets looks like.
- **Losses vastly outnumber gains.** Per question, against the baseline:

  | Run | Base right → FT wrong | Base wrong → FT right |
  |---|---|---|
  | LoRA r=16 (seeds 1, 2, 3) | 342 / 346 / 356 | 43 / 42 / 40 |
  | LoRA r=8 / r=32 | 361 / 361 | 41 / 43 |
  | QLoRA r=16 (seeds 1, 2, 3) | 396 / 392 / 384 | 38 / 36 / 35 |
  | QLoRA r=8 / r=32 | 398 / 393 | 27 / 28 |

  Fine-tuning fixes about 40 questions but breaks about 350–400 that the base model already got right.
- **Rank barely matters.** r=8, 16 and 32 land within about 1.5 points of each other. More adapter capacity does not recover the lost reasoning, which fits a problem with the training data rather than with capacity.

**Caveat.** The final-eval records (`eval_records.json`) store the extracted answer but not the generated text, so we have not directly confirmed that the fine-tuned models write shorter answers. The explanation rests on the training data, the timing of the drop and the per-question pattern above. Saving `generated_text` in the eval records is the first follow-up item (§9).

### 7.4 Secondary factors (likely smaller)

- **Loss on prompt tokens.** `DataCollatorForLanguageModeling(mlm=False)` trains on the whole sequence (chat template, instruction and question), not just the answer. It spends adapter updates on reproducing questions.
- **Aggressive learning rate.** 2e-4 for 3 epochs is a typical LoRA setting for teaching a model a new task, but it is strong for one that is already at 89% on that task.

---

## 8. LoRA vs QLoRA

On this hardware, QLoRA is worse on every axis:

| | LoRA r=16 (seeds 2–3) | QLoRA r=16 | QLoRA vs LoRA |
|---|---|---|---|
| Accuracy | 65.3% | 61.9% | −3.4 pt |
| Training energy | 0.269 kWh | 0.313 kWh | **+16%** |
| Training time | 1.79 h | 2.18 h | +22% |
| Peak GPU memory | 14.2 GB | 10.7 GB | −25% |

QLoRA's 4-bit weights save memory, but each forward and backward pass must dequantize them, which costs time and so energy. Its only advantage, lower memory, did not matter here because LoRA already fit on the 24 GB A10G. **QLoRA is only the greener choice when it lets you use a smaller GPU or a larger batch.** Using it when LoRA already fits costs both energy and accuracy. Average GPU power was nearly the same (130 W vs 138 W), so the extra energy comes from running longer, not from drawing more power.

Adapter rank had no measurable effect on energy (LoRA r=8/16/32: 0.275 / 0.269 / 0.267 kWh), as expected: adapter parameters are a tiny fraction of the per-step compute, which is dominated by the frozen 5B-param base and the 262k-vocabulary logits.

---

## 9. Conclusions

1. **Fine-tuning a strong instruction-tuned model on GSM8K's reference solutions reduced accuracy by 23–28 points, at a cost of 0.27–0.32 kWh (0.10–0.12 kg CO2e) per run.** Every REI is negative (about −0.87 /kWh; about −2.3 /kg CO2e). In Green Gap terms, the gap is at zero: no amount of energy in this setup bought improvement, because the first 20% of training had already lowered accuracy.
2. **Whether fine-tuning pays off depends on the starting point and the data, not just the method.** Supervised data whose reasoning style is weaker than the model's own makes the model worse however efficiently it is applied.
3. **QLoRA is not automatically greener than LoRA.** Without a memory constraint it used 16% more energy and scored 3.4 points lower.
4. **Adapter rank (8–32) changes neither accuracy nor energy meaningfully** in this regime.

---

## 10. Limitations

- Full Fine-Tuning (Condition A) has not been run, so the three-way comparison is incomplete.
- One model, one benchmark, one GPU type, one region.
- Energy covers training only; final-eval energy (about 50 minutes per run) is excluded by design.
- Mid-training curves use 100 questions, so each point carries about ±5 points of sampling noise.
- `lora_r16_seed1` energy is inflated by the one-at-a-time mid-training evals (§6.4).
- The baseline used a 786-token budget instead of 768 (bounded effect of ≤0.7 points, §4).
- The style-imitation explanation is inferred, not yet confirmed from saved generations (§7.3).

---

## 11. Future work

1. **Train on the model's own correct answers (self-distillation / rejection sampling).** Generate answers with the base model for the 7,473 training questions, keep only the correct ones (about 89%), and fine-tune on those. The targets then match the model's own reasoning style instead of replacing it, so a real gain above 88.8% becomes possible. Generating the data takes about 4–5 h on the A10G, and that energy must be counted in the method's REI, since it is part of the method's cost.
2. **Switch to the non-instruction-tuned base model (`google/gemma-4-E2B`).** A pretrained model that is not instruction-tuned should have a much lower zero-shot accuracy, so GSM8K fine-tuning should produce a large positive Δ. Its baseline has not been measured yet. This is the classic setting for measuring "energy per point gained". It needs a new baseline and a full re-run of the matrix.
3. **Low-cost follow-ups to this study:** save `generated_text` in `eval_records.json` to confirm the style-imitation explanation directly; compute loss on the answer only; strip `<<…>>` calculator markup from targets; try a lower LR or a single epoch; rerun `lora_r16_seed1` for a clean energy figure; re-measure the baseline at 768 tokens; measure final-eval energy too.
4. **Run Condition A (Full FT)** on g6e.xlarge (L40S, 48 GB) to complete the comparison.

---

## 12. Where everything is

| Path | Contents |
|---|---|
| `results-g5-backup/metrics.csv` | One row per run: accuracy, energy, CO2e, REI |
| `results-g5-backup/emissions/emissions.csv` | Every CodeCarbon session, including memory-gate runs |
| `results-g5-backup/runs/<method>/<run_id>/result.json` | Per-run summary |
| `results-g5-backup/runs/<method>/<run_id>/eval_records.json` | Per-question predictions for the final eval |
| `results-g5-backup/runs/<method>/<run_id>/checkpoint_curve.csv` | Green Gap curve points |
| `results-g5-backup/runs/<method>/<run_id>/final_adapter_or_model/` | Final adapters (can be re-evaluated on any GPU) |
| `results-g5-backup/verify_final2_tmux_output.txt` | Console tail of the last run (`qlora_r32_seed1`) |
| `baseline_accuracy.json`, `baseline_records.jsonl` | Zero-shot baseline, with generated text |
