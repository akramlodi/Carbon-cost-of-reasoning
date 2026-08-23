"""Model/tokenizer loading and LoRA/QLoRA adapter setup.

A single `load_model(config)` entrypoint drives all three conditions --
full fine-tuning, LoRA, and QLoRA -- based on the `quantization` and `lora`
blocks of the run config, so the training script doesn't need per-method
branching.
"""
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.get("gradient_checkpointing", False)
        )
    elif config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    lora_cfg = config.get("lora", {}) or {}
    if lora_cfg.get("enabled"):
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model
