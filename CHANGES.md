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
- [ ] **`--method full_ft`** — not run yet; needs Change 1's implementation exercised on a GPU. On a free T4, expect this may still hit OOM during the actual training step, since the full 5.1B-param model must still be loaded unquantized even with most of it frozen — that's expected and consistent with the README's two-phase compute plan (T4 for pipeline validation, rented GPU for the real Full FT run). The useful checks on T4 are: (a) the trainable-parameter count printed after freezing is correct (~1.87B, not 5.1B), and (b) it gets past model loading and into the training step before any OOM, confirming the freeze logic itself works even if the run can't complete here.
- [x] **`--method qlora`** — ran and passed (10 steps, loss 1.5→1.2, adapter save/reload OK). Change 1's freezing logic only touches the `full_ft` branch of `load_model()`, so this result stands even though the file was edited afterward to add it.

---

## Resolution (post-implementation)

- **Change 1 — implemented.** `frozen_modules` added to `configs/full_ft.yaml` (best-guess prefixes, explicitly flagged as needing verification against the real checkpoint's `named_parameters()` on first GPU run — no code change needed to correct them). Freezing logic + trainable-param logging added to `src/models.py` (`_freeze_non_backbone_params`, `_log_trainable_parameters`), wired into the non-quantized/non-LoRA branch of `load_model()`. README's Condition A section updated with the rationale.

- **Also fixed along the way** (not from this doc, surfaced during the `--method lora` run above): `PeftModel.from_pretrained` in the adapter-reload sanity check was failing with `KeyError: '...k_norm'` because the first model was still resident on the GPU when a second copy was loaded, forcing `device_map="auto"` to partially offload it — and PEFT has a bug loading an adapter onto a partially-offloaded base model. Fixed in `scripts/smoke_test.py` by freeing the first model (`del trainer, model; gc.collect(); torch.cuda.empty_cache()`) before loading the second copy.