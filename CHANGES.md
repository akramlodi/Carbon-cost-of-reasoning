# CHANGES.md

Two decisions made during smoke-test debugging that need to be implemented / documented in the codebase. Both are described with context so the reasoning is auditable later, not just the diff.

---

## Change 1: Full Fine-Tuning parameter scope must exclude embedder + multimodal encoders

### Context
`google/gemma-4-E2B` loads as **5,105,636,896 total parameters** — not the "2.3B effective" figure used in Google's marketing. The gap is made up of:
- Embedder (incl. Per-Layer Embeddings): ~2.74B params
- Language backbone ("Einsums" — attention + MLP transformer layers): ~1.87B params
- Audio encoder: 305M
- Vision encoder: 150M
- Speculative-decoding drafter: 76M

### Decision
For the **Full Fine-Tuning** condition, freeze the audio encoder, vision encoder, drafter, and embedder. Only the language backbone (attention + MLP layers, ~1.87B params) is trainable.

### Rationale
1. **Architecturally consistent, not an arbitrary restriction.** Google's Gemma 4 technical report states the audio and vision encoders are kept frozen even during the base model's own pretraining. GSM8K is text-only, so there is no gradient path to these components regardless — training them would only consume memory and compute for zero signal.
2. **Fair comparison across the three methods.** LoRA and QLoRA (as configured in `configs/lora.yaml` and `configs/qlora.yaml`) only adapt attention-projection modules inside the backbone — they never touch the embedder. If Full FT is allowed to also update ~2.74B embedding params, the study is no longer isolating "fine-tuning method" as the variable — it's conflating method with scope, which undermines the accuracy/energy comparison this whole project is built on.
3. **Matches the research question.** The paper is about specializing a model for reasoning; that capability lives in the transformer backbone, not in token embeddings, and GSM8K introduces no new vocabulary.
4. **Keeps the compute budget sane.** ~1.87B trainable params (vs. 5.1B) brings Full FT's GPU requirement back to roughly A100 40GB rather than needing an 80GB card.

### Implementation instructions
- In `src/models.py`, in the Full FT branch of `load_model()`, after the model loads: iterate `model.named_parameters()` and set `requires_grad = False` for any parameter whose name matches the audio encoder, vision encoder, drafter, and embedder submodules.
  - **Do not assume the exact attribute prefixes** (`audio_encoder.`, `vision_encoder.`, `drafter.`, `embedder.`) — confirm the real names first by printing `[n for n, _ in model.named_parameters()]` on the loaded checkpoint, since Gemma 4's actual module naming may differ from this shorthand.
- After freezing, log the resulting trainable parameter count in the same style as the existing `print_trainable_parameters()` output used on the LoRA/QLoRA paths, so Full FT run logs show the same sanity-check line (e.g. `trainable params: X || all params: 5,105,636,896 || trainable%: Y`).
- Add a `frozen_modules` list to `configs/full_ft.yaml` rather than hardcoding the freeze logic silently in `models.py` — this makes the scope decision visible/auditable in the config itself, and allows toggling it later as an ablation (e.g. "what if the embedder is also unfrozen?") without touching code.
- Update the README's Methodology section (Condition A — Full Fine-Tuning) to state this frozen scope explicitly, since it's now a methodological decision that belongs in the paper's writeup, not just an implementation detail.

---

## Verification / smoke test plan

- [x] **`--method lora`** — ran and passed (10 steps, loss 1.44→1.17, adapter save/reload OK) after also fixing an unrelated GPU-double-residency bug in the adapter-reload check (see below).
- [x] **Full FT `frozen_modules` scope** — verified against the real checkpoint via `scripts/check_full_ft_scope.py` (load + freeze + inspect, no training steps, safe on free T4). Two rounds on `google/gemma-4-E2B` (base): the first confirmed the overall trainable ratio (36.03%, close to the ~1.87B/5.1B estimate) but also surfaced dead patterns (`vision_encoder`, `audio_encoder`, `drafter` all matched nothing). A full-checkpoint keyword scan (not just a truncated name dump) on the second round confirmed: no speculative-decoding drafter exists anywhere in this checkpoint (`'draft'`/`'speculat'` both 0 matches); the real names are `vision_tower`/`audio_tower` (not `*_encoder`); and two small multimodal fusion projections (`model.embed_vision.embedding_projection`, `model.embed_audio.embedding_projection`) were being missed entirely. Also found the original `per_layer` pattern was over-broad — it froze not just the Per-Layer Embedding table (already covered by `embed_tokens` as a substring) but also `per_layer_model_projection`/`per_layer_projection_norm`, which are backbone forward-pass integration logic, not embedder, and should stay trainable. Final `frozen_modules`: `vision_tower`, `embed_vision`, `audio_tower`, `embed_audio`, `embed_tokens` — yields 1,877,105,920 / 5,104,297,504 trainable (~36.78%), matching the ~1.87B estimate even more closely than the first pass. A third round after the project switched to `google/gemma-4-E2B-it` (see below) reran the same script against the new checkpoint and got byte-identical parameter names, counts, and the same 36.78% ratio — confirming `-it` is the same architecture with different weights, so this scope carries over with no changes needed.
- [ ] **`--method full_ft` real training run** — not run yet, and deliberately won't be smoke-tested on T4 (README already documents the expected OOM there: the full 5.1B raw params must still load unquantized even with most frozen). First real exercise of this path is the actual rented-A100 run.
- [x] **`--method qlora`** — ran and passed (10 steps, loss 1.5→1.2, adapter save/reload OK). Change 1's freezing logic only touches the `full_ft` branch of `load_model()`, so this result stands even though the file was edited afterward to add it.

---

## Resolution (post-implementation)

- **Change 1 — implemented.** `frozen_modules` added to `configs/full_ft.yaml` (best-guess prefixes, explicitly flagged as needing verification against the real checkpoint's `named_parameters()` on first GPU run — no code change needed to correct them). Freezing logic + trainable-param logging added to `src/models.py` (`_freeze_non_backbone_params`, `_log_trainable_parameters`), wired into the non-quantized/non-LoRA branch of `load_model()`. README's Condition A section updated with the rationale.

- **Also fixed along the way** (not from this doc, surfaced during the `--method lora` run above): `PeftModel.from_pretrained` in the adapter-reload sanity check was failing with `KeyError: '...k_norm'` because the first model was still resident on the GPU when a second copy was loaded, forcing `device_map="auto"` to partially offload it — and PEFT has a bug loading an adapter onto a partially-offloaded base model. Fixed in `scripts/smoke_test.py` by freeing the first model (`del trainer, model; gc.collect(); torch.cuda.empty_cache()`) before loading the second copy.

---

## Change 3 (not from this doc, decided later in the same session): switch to the instruction-tuned checkpoint + chat-template prompting

### Decision
`model_id` switched from `google/gemma-4-E2B` (base) to `google/gemma-4-E2B-it` (instruction-tuned) everywhere it's used for training/eval (`configs/full_ft.yaml`, `configs/lora.yaml`, `configs/qlora.yaml`, `scripts/smoke_test.py`, `scripts/measure_baseline.py`). Prompt construction switched from a raw completion-style string (`"Question: ...\nAnswer: ..."`) to `tokenizer.apply_chat_template()` (`src/data.py::format_example`), since an instruction-tuned model expects its own turn-formatted input to perform as intended.

### Why
The base checkpoint isn't instruction-tuned, so a raw completion prompt is closer to what it actually expects; but the project's zero-shot baseline measurement surfaced the question of whether to measure the base or `-it` checkpoint, and `-it` was chosen for consistency between the baseline and the fine-tuning conditions it's compared against. Once on `-it`, using the raw completion prompt would under-measure the model's real capability (it wasn't trained to expect that format), so the chat template followed as a consequence.

### Side-effect fixed
`tokenizer(text, add_special_tokens=True)` (the default) on an already chat-templated string double-adds the BOS token, since the template embeds it itself. Both re-tokenization sites (`src/data.py::tokenize_dataset` for training, `src/evaluate.py::generate_answer` for eval) now pass `add_special_tokens=False`.

### Verification
`scripts/check_full_ft_scope.py` was rerun against `-it` after the switch (see the Full FT `frozen_modules` scope entry above) — identical results to the base checkpoint, confirming the architecture (and therefore the frozen-scope patterns) is unaffected by the checkpoint swap. The LoRA/QLoRA smoke tests have **not** been rerun since the chat-template change yet — that's the next thing to verify, since it changes the actual token sequences the model trains/evaluates on, unlike the checkpoint swap which didn't.

---

## Change 4: terse final-answer instruction added to the prompt

### Context
A qualitative check via `scripts/inspect_generations.py` (5 examples) showed every generation cutting off mid-sentence or mid-calculation before stating a final answer ("Final Value = $8", "= 9", "(Total") -- except the two shortest problems, which completed fully and scored correctly. The model was writing essay-style, hedging explanations that ran past the 256-token generation budget before ever reaching an answer.

### Decision
`INSTRUCTION` in `src/data.py` gained a terse-answer directive: `"...Let's think step by step. End your response with only: #### <number>"`. Since `format_example` builds both the eval-time prompt and the training-time text from the same `INSTRUCTION`, this applies to both -- and for training, the target already naturally complies (GSM8K reference solutions already end in `#### <answer>`), so there's no mismatch introduced between what the model is told to do and what it's shown as the correct target.

### Rationale
Two effects from one change: (1) gives `extract_final_answer` a reliable anchor to key off instead of falling back to "last number in the text" when generation is truncated mid-calculation, and (2) discourages the essay-style hedging that was consuming the token budget, so more generations reach a stated answer before hitting `max_new_tokens`.

### Impact on prior results
This invalidates the 50-example zero-shot baseline measured before this change (`results/baseline_accuracy.json`, accuracy 0.10) -- it was measured under the old prompt. Re-run `scripts/measure_baseline.py` (quick 50-sample check first, then the full set) before using a baseline number for real REI calculations. The LoRA/QLoRA smoke tests are similarly stale against this prompt version, on top of already being stale against the chat-template switch in Change 3 -- both should be re-verified together in the next smoke test run rather than separately.

---

## Change 5: LoRA/QLoRA OOM at step 0 on g5.xlarge (24GB) -- logits-tensor memory

### The failure
`GG_METHODS=lora,qlora ./scripts/run_matrix.sh` OOM'd on `g5.xlarge` (A10G, ~22GB usable) at step 0 of 1,404, before any training step completed and before the first periodic eval (which wouldn't fire until ~step 281):

```
File ".../transformers/models/gemma4/modeling_gemma4.py", line 2583, in forward
    logits = logits / final_logit_softcapping
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.21 GiB.
GPU 0 has a total capacity of 22.06 GiB, ... this process has 20.90 GiB memory in use.
```

This was the LoRA config (the first method in the matrix). Distinct from the earlier eval-callback leak -- that code is never reached before crashing here.

### Root cause
Gemma-4 uses a **262,144-token vocabulary**, so the final logits tensor is `per_device_batch_size × seq_len × 262144`, and the softcapping divide allocates a second copy of it (plus `float32` upcasting in the loss path multiplies the cost by ~2x vs bf16). The traceback's 1.21 GiB allocation is exactly `8 × ~155 × 262144 × 4 bytes` -- i.e. the failing batch was only ~155 tokens long, but already needed 1.2 GiB per fp32 logits copy. On top of ~10.2GB of bf16 weights + the activations of 8-example batches with **gradient checkpointing off**, that exceeded the card.

Two configuration gaps let this through: `configs/lora.yaml` had `per_device_train_batch_size: 8` and `gradient_checkpointing: false`. Neither showed up in the earlier small-scale smoke tests (fewer steps, and the *_E2B* base/draft shrank the visible footprint enough to pass).

### What actually had to change (measured, not guessed)
1. **`per_device_train_batch_size`: 8 -> 2, `gradient_accumulation_steps`: 2 -> 8** (both `lora.yaml` and `qlora.yaml`). Effective batch stays **16**, matched across all three conditions as before. Because the dataloader walks the same examples in the same order and gradient accumulation sums 8×2 = 2×8 consecutive micro-batches, optimizer steps receive **identical data** -- step count stays essentially unchanged (1,404 with ceil) and comparability across conditions is preserved. This is the direct fix: it divides the logits tensor by 4.
   - Applied to QLoRA too, even though QLoRA's weights are 4-bit: quantization shrinks the **weights**, not the **activations/logits tensor**, which is byte-identical in size under both methods.
2. **`gradient_checkpointing: true` in `configs/lora.yaml`** (QLoRA already had it). This was genuinely off in the code path that OOM'd -- the flag was supported by `build_training_args()` but not enabled for LoRA.
   - Required a companion fix so it actually trains: `src/models.py::load_model` now calls `model.enable_input_require_grads()` on the non-quantized gradient-checkpointing path (the LoRA branch), otherwise reentrant checkpointing sees no input requiring grad and silently stops updating the adapters. The 4-bit/QLoRA path (`_freeze_base_model`) already did this.
3. **`max_seq_length` (512): NOT changed, and confirmed not the problem.** Measured the real tokenized training set (7,473 examples, exactly as `src/data.py` builds it -- chat template + `INSTRUCTION`): mean 216, p50 203, p90 310, p99 425, max 571. Only 7/7473 (0.09%) exceed 512; reducing to 384 would truncate 2.1% of training targets (and drop the `#### answer` tails -- the very signal being trained). Additionally, `DataCollatorForLanguageModeling` pads only to the **longest example in each batch**, not to `max_seq_length`, so the cap never inflates the logits tensor -- the failing batch was ~155 tokens. Leave it at 512.
4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: was missing from the real entrypoint, now set there.** `scripts/measure_baseline.py` (`os.environ.setdefault`) had it, but the training path did not. Added the same `setdefault` at the top of `src/train.py`, before any CUDA use -- this module is the entrypoint behind `scripts/run_experiment.py` and `scripts/run_matrix.sh`, so a single site covers the real matrix. This doesn't fix the underlying requirement; it removes fragmentation OOMs on top of it.

### Diagnostic additions
`GreenGapCheckpointCallback.on_step_end` now prints `[MEM] allocated/reserved/peak` every 10 steps, so a live run (or an intermediate-scale check) shows whether GPU memory is climbing toward the ceiling rather than discovering it at a crash.

### Intermediate-scale gate before the full matrix
Do **not** jump from this fix straight to the 1,404-step run. Cheapest meaningful check, larger than the 64-sample smoke test but far cheaper than the matrix:

```
python -m src.train --config configs/lora.yaml --seed 999 \
  --n_train_samples 200 --n_eval_samples 32
```

(200 samples x 3 epochs x effective-16 = ~37 optimizer steps; the `--n_eval_samples 32` matters because the trailing eval in `src/train.py::run` iterates `evaluate_model` one example at a time with `max_new_tokens=768` -- a full 1,319-example eval would add ~1-2 hours to what is nominally a memory check. `python -m src.train` (not `run_experiment.py`) deliberately avoids writing a `results/metrics.csv` row for seed 999, and the seed-999 output dir is never consulted by `run_matrix.sh`'s skip logic for seeds 1-3.)

Watch the `[MEM]` lines: they should plateau well under ~20GB. Only when that passes cleanly, run:

```
GG_METHODS=lora,qlora GG_BASELINE_ACCURACY=0.8877937831690674 \
  GG_GPU_MODEL=A10G-24GB GG_REGION=us-east-1 ./scripts/run_matrix.sh
```

### Applies to `configs/full_ft.yaml` too? (for the g6e.xlarge run later)
Same findings transfer: full_ft runs `per_device_train_batch_size: 4` on a 48GB L40S today, with `gradient_checkpointing: true` and `max_seq_length: 512` already set. Its bf16 weights (~10.2GB on a 48GB card) leave far more headroom, and its `4 × ~155 × 262144 × 4B ≈ 0.65 GiB` logits copies are only ~1/2 of LoRA's at the same scale -- but the logits math above (and the `enable_input_require_grads` requirement) applies identically if full_ft ever needs batch cooldown. `expandable_segments` is inherited automatically since it's set in `src/train.py`, which full_ft also runs through.
---

## Change 6: `evaluate_model()` batched -- the end-of-run eval was the real cost, not training

### Context
The first real `run_matrix.sh` run (`configs/lora.yaml`, seed 1) finished all 1,404 training steps, then sat in the trailing `evaluate_model()` call for over an hour at ~30% GPU utilization. `src/evaluate.py::evaluate_model` looped the dataset one example at a time, and the trailing eval in `src/train.py::run` scores all 1,319 GSM8K test examples with `max_new_tokens=768`. Batching is not a micro-optimization here: one example at a time means one decode stream, so the GPU is mostly idle waiting on the per-token Python/launch overhead, and a batch of 16 amortizes that across 16 rows.

This is a per-run tax paid at the end of **every** run in the matrix -- 6 core LoRA/QLoRA runs + 4 rank ablations on g5.xlarge, plus the 3 full_ft runs later on g6e.xlarge -- so the fix moves from "one slow run" to "11 hours across the matrix" (plus 3 more on the L40S).

`scripts/measure_baseline.py` had already hit this exact wall measuring the zero-shot baseline and solved it with a `generate_batch()` helper. Rather than write a second implementation, that helper moved into `src/evaluate.py` and `measure_baseline.py` now imports it, so the baseline measurement and the training-time eval cannot drift apart. `--batch_size` there defaults to `src.evaluate.EVAL_BATCH_SIZE` rather than a second hardcoded 16.

### What changed
- **`src/evaluate.py::generate_batch()`** (ported from `measure_baseline.py`): left-padded batched generation under `torch.no_grad()`, with per-row EOS scanning. Left-padding is what makes a single shared `input_ids.shape[1]` slice index legal; per-row EOS scanning is still required because `generate()` pads finished rows out to `max_new_tokens`, so a row's real length is not readable from any shared value. `normalize_eos_ids()` moved over too -- Gemma stops on more than one token (`<eos>` and `<end_of_turn>`), so the scan has to test a set, not an int.
- **`src/evaluate.py::evaluate_model()`**: now iterates the dataset in batches and takes a `batch_size` argument. `generate_answer()` (the unbatched path) is kept deliberately, as the oracle the batched path is tested against.
- **`padding_side` is set and restored inside `generate_batch`**, not at the call site. The training path shares one tokenizer between eval and the trainer's data collator, so permanently flipping its padding side would left-pad *training* batches too (Gemma computes position ids from a plain `arange`, so that would shift RoPE positions for real tokens).

### Two call sites, two batch sizes
This is the second time a memory problem has come out of `evaluate_model()` being called mid-training (the first was the missing `no_grad`, fixed earlier), so the batch size is an explicit argument at each call site rather than one shared default:

| Call site | Examples | Batch | Why |
| --- | --- | --- | --- |
| `src/train.py::run` (end of run) | 1,319 | `EVAL_BATCH_SIZE` = 16 | Training is over; the same size `measure_baseline.py` measured the baseline at on this exact card. Back off to 8/4 on OOM. |
| `GreenGapCheckpointCallback` (mid-training) | 100 (`n_eval_subsample`) | `MID_TRAIN_EVAL_BATCH_SIZE` = 4 | Runs against a model that still holds optimizer state and resumes training immediately, on the card where a 262k-vocab logits tensor plus its softcapping copy already OOM'd at `per_device_train_batch_size: 8` (Change 5). |

Under `no_grad` the eval's incremental cost is mostly the KV cache, not activations, so batch 4 is conservative rather than a measured minimum -- but the callback has no headroom to spare and no reason to ask for 16 rows of it. A test asserts the mid-training default stays below the standalone one, so the two can't silently converge.

### Verification
- [x] **Local, CPU** (`tests/test_evaluate_batching.py`, 9 passed): batched vs. per-example parity, checked two ways. A stub model with a scripted `generate()` pins the semantics the batching depends on (shared slice index under left-padding, per-row EOS length, multi-token EOS set, row ordering, `no_grad`, train-mode restore). A tiny real HF causal LM on the project's real gemma tokenizer checks that batched greedy generation reproduces the unbatched `generate_answer()` text example-for-example across deliberately uneven prompt lengths. Each of those tests was mutation-checked: flipping to right-padding, dropping the per-row EOS scan, dropping `no_grad`, and mispairing examples with generations each fail at least one test.
- [ ] **GPU, g5.xlarge** -- still required before the matrix, and *not yet run*. `scripts/verify_batched_eval.py` does checks 1 and 2 in one command (real-checkpoint parity against the unbatched path, plus timing both sides and extrapolating to the full 1,319-example eval; exits non-zero on any mismatch or if batching isn't faster):
  ```
  python scripts/verify_batched_eval.py \\
    --adapter results/runs/lora/lora_r16_seed1/checkpoint-1404
  ```
  Check 3 is the Change 5 memory gate, re-run because the mid-training eval is now batched rather than one-at-a-time, so its memory profile is genuinely new even though its peak should stay far below the training peak under `no_grad`:
  ```
  python -m src.train --config configs/lora.yaml --seed 999 --n_train_samples 200 --n_eval_samples 32
  ```
  Confirm `[MEM]` still plateaus across the mid-training evals. (`save_strategy="epoch"` means `checkpoint-1404/adapter_model.safetensors` exists even though the interrupted run never reached `final_adapter_or_model/` -- it's written after the trailing eval.)

### Batched vs unbatched predictions are not identical, and that is expected
The GPU check on the `lora_r16_seed1` checkpoint reported 2/32 examples where the batched path predicted a different number than the unbatched path -- and in opposite directions (index 13: unbatched 12.0 wrong, batched 13.0 right; index 9: 135.0 vs 180.0, both wrong). This is **not** a slicing/padding bug, and it is worth being precise about why, because "the eval disagrees with itself" would otherwise be a fatal-sounding result for the whole matrix.

Greedy decoding is discontinuous and the arithmetic underneath it is not batch-invariant. A different batch size means different matmul shapes, so cuBLAS picks different kernels and reduces in a different order, which perturbs logits at roughly the 1e-3 level. Wherever the top two candidates are that close, the argmax flips, one token differs, and from that point the two continuations are different sentences that can land on different final numbers. A 262k-vocab model writing hundreds of tokens of arithmetic is exactly where near-ties live.

Four things were checked rather than assumed:

- **Left-padding is handled correctly for this architecture.** `generate()` derives `position_ids` from the attention mask (`cumsum - 1`, pad positions zeroed, `generation/utils.py::_prepare_position_ids_for_generation`), so a left-padded row's real tokens still occupy positions 0..n-1. Gemma-4's Per-Layer Embeddings are keyed on **token identity**, not position (`embed_tokens_per_layer(input_ids)`, see `get_per_layer_inputs`), so pad prefixes cannot shift them. Attention masks exclude pad positions, and RoPE is relative, so a constant per-row offset cancels. The shared slice index is sound.
- **The disagreement survives with no padding at all.** A 1-row batch has zero left-padding, and it disagrees with the 16-row batch on some of the same examples (`scripts/diagnose_batched_divergence.py`, Check D). That rules out the padding/slicing explanation and points at arithmetic order.
- **Batch composition has no effect at all.** A prompt's generation is byte-identical regardless of which other examples shared its batch. This is the invariant Check E measures -- and Check E as first written was measuring nothing: it compared `window0[i]` against `window1[i]` where window 1 was `prompts[1:]`, so every comparison was silently between two *different* questions, which is why its values looked like window 0's shifted by one. Read correctly (window 1's value at `i` equals window 0's at `i+1`, i.e. the same prompt), all 15 overlapping examples matched exactly. `scripts/diagnose_batched_divergence.py` now keys results by 1-based example index instead of batch position so this cannot recur, and `tests/test_evaluate_batching.py` pins the invariant on CPU.
- **Only the batch *size* matters, and only for the extracted answer.** Going from 16 rows to 1 row changed the generated text on 11 of 16 examples but the final answer on only 2 -- two very different rates, and the answer-level one is what REI depends on. The first differing token sat at generation steps 79-247 with top1-top2 logit gaps of 0.0000 and 0.1250: exact near-ties, as predicted.

**What the experiment actually needs is reproducibility, not byte-equality with a path that is no longer used.** The batched path is deterministic: the identical command twice produces identical predictions (Check C in `scripts/verify_batched_eval.py`, which now gates on this), because batches are contiguous deterministic slices of the test set. Two further points make the comparison sound:

1. **The zero-shot baseline (0.8877937831690674) was already measured batched at 16** by `scripts/measure_baseline.py`, in the same contiguous order. So after this change, baseline and every fine-tuned condition share identical batch boundaries. Before this change they did *not* -- the baseline was batched while the evals were unbatched, which injected the same class of per-example flip noise directly into the REI numerator. Batching removes that inconsistency rather than adding one.
2. **The flips are not directional.** They move examples both ways, which is why the gate reports wrong->right and right->wrong counts and the accuracy delta, not just a mismatch count. At ~6% of examples, the plausible effect on a 1,319-example accuracy is a few tenths of a percentage point in either direction, non-systematic, and identical in distribution across conditions.

`scripts/verify_batched_eval.py` therefore fails on a mismatch *rate* above `--max_mismatch_rate` (default 0.10) or under `--strict`. Because a couple of flipped answers move a small sample's rate by several points -- the same 16 examples read 12.5% here and 6.25% over 32 -- the script now warns below `--n_examples 64` and the gate should be judged on a full 128-example run, not on these samples., and always fails if batching is slower or if the batched path is not reproducible. Per-example disagreement below that rate is reported with the explanation above rather than blocking the matrix. **If a future run shows a mismatch rate well above 10%, or one that moves accuracy consistently in one direction, treat it as a real bug and use `scripts/diagnose_batched_divergence.py` before spending more instance time.**
