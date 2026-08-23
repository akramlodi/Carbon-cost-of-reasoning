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