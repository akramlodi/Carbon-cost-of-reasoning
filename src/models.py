"""Model/tokenizer loading and LoRA/QLoRA adapter setup.

A single `load_model(config)` entrypoint drives all three conditions --
full fine-tuning, LoRA, and QLoRA -- based on the `quantization` and `lora`
blocks of the run config, so the training script doesn't need per-method
branching.
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _freeze_base_model(model, use_gradient_checkpointing=False):
    """Stand-in for peft.prepare_model_for_kbit_training, minus its blanket
    fp32 upcast of every non-4bit-quantized param.

    That upcast is fine for typical dense transformers, but gemma-4-E2B's
    "effective 2.3B" footprint relies on multi-GB per-layer embedding
    buffers that bitsandbytes never quantizes (it only replaces nn.Linear);
    those stay resident in bf16, and upcasting them to fp32 in one shot can
    exceed a 16GB GPU's headroom. Since this pipeline trains in bf16
    throughout (bnb_4bit_compute_dtype=bfloat16), that upcast isn't needed
    for numerical stability here -- just freeze the base model and wire up
    gradient checkpointing.
    """
    for param in model.parameters():
        param.requires_grad = False
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def _log_trainable_parameters(model):
    """Mirrors peft's own print_trainable_parameters() output format, for a
    plain (non-peft-wrapped) model -- used on the Full FT path so run logs
    are consistent across all three conditions."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")


def _freeze_non_backbone_params(model, frozen_module_patterns):
    """Full FT should only train the language-model transformer backbone,
    not the full raw checkpoint (audio/vision encoders, speculative drafter,
    embedder) -- see configs/full_ft.yaml for the rationale. Freezes any
    parameter whose dotted name contains one of frozen_module_patterns.
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in frozen_module_patterns):
            param.requires_grad = False
    _log_trainable_parameters(model)
    return model


def load_model(config):
    model_id = config["model_id"]
    quant_cfg = config.get("quantization", {}) or {}
    kwargs = {"device_map": "auto"}

    if quant_cfg.get("enabled"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if config.get("bf16", True) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if quant_cfg.get("enabled"):
        model = _freeze_base_model(
            model, use_gradient_checkpointing=config.get("gradient_checkpointing", False)
        )
    elif config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    lora_cfg = config.get("lora", {}) or {}
    if lora_cfg.get("enabled"):
        lora_kwargs = dict(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            task_type="CAUSAL_LM",
        )
        # gemma-4's attention/MLP Linear layers are wrapped in
        # Gemma4ClippableLinear (input/output clamping for numerical
        # stability); an explicit target_modules list matches that wrapper
        # directly and peft<0.19 can't inject LoRA into it (see
        # https://github.com/huggingface/peft/issues/3129). peft>=0.19 ships
        # Gemma-4-aware default target modules that scope correctly to the
        # LM layers and avoid the vision/audio ClippableLinear modules -- so
        # only pass target_modules when the config explicitly overrides it.
        target_modules = lora_cfg.get("target_modules")
        if target_modules:
            lora_kwargs["target_modules"] = target_modules
        peft_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif config.get("frozen_modules"):
        model = _freeze_non_backbone_params(model, config["frozen_modules"])

    return model
