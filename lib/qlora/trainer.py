"""
QLoRA Training Module
Adapted from: https://github.com/artidoro/qlora
"""

import os
import logging
from os.path import exists, join, isdir
from typing import Optional, Dict

import torch
import bitsandbytes as bnb
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from peft.tuners.lora import LoraLayer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def find_all_linear_names(model, bits: int = 4) -> list:
    """
    Find all linear layer names in the model that should have LoRA applied.

    For quantized models, we look for bnb.nn.Linear4bit or Linear8bitLt layers.
    For full precision, we look for torch.nn.Linear layers.

    Args:
        model: The model to search
        bits: Number of bits for quantization (4, 8, or 16/32 for full precision)

    Returns:
        List of layer names to apply LoRA to
    """
    if bits == 4:
        cls = bnb.nn.Linear4bit
    elif bits == 8:
        cls = bnb.nn.Linear8bitLt
    else:
        cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Remove lm_head as it's typically not trained with LoRA
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    """
    Callback to save PEFT adapter weights separately during training.
    This allows for efficient checkpoint saving without duplicating the base model.
    """

    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # Remove the full pytorch_model.bin as we only need the adapter
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        # Also remove model.safetensors if present
        safetensors_path = os.path.join(checkpoint_folder, "model.safetensors")
        if os.path.exists(safetensors_path):
            os.remove(safetensors_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and model embeddings when adding special tokens.

    Note: This is the unoptimized version that may make embedding size not divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        # Initialize new token embeddings with the average of existing embeddings
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_qlora_model(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    bits: int = 4,
    double_quant: bool = True,
    quant_type: str = "nf4",
    lora_r: int = 64,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
    use_gradient_checkpointing: bool = True,
    trust_remote_code: bool = True,
    max_memory_mb: int = 80000,
    checkpoint_dir: Optional[str] = None,
    fp16: bool = False,
    bf16: bool = True,
    hf_token: Optional[str] = None,
):
    """
    Load a model with QLoRA configuration.

    Args:
        model_name_or_path: HuggingFace model name or local path
        cache_dir: Cache directory for model weights
        bits: Quantization bits (4 or 8)
        double_quant: Use double quantization for memory efficiency
        quant_type: Quantization type ("nf4" or "fp4")
        lora_r: LoRA rank dimension
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Specific modules to apply LoRA to (None = auto-detect)
        use_gradient_checkpointing: Enable gradient checkpointing to save memory
        trust_remote_code: Trust remote code from HuggingFace
        max_memory_mb: Maximum GPU memory to use per device
        checkpoint_dir: Path to existing PEFT checkpoint to resume from
        fp16: Use float16 precision
        bf16: Use bfloat16 precision
        hf_token: HuggingFace auth token for accessing private/gated models

    Returns:
        tuple: (model, tokenizer)
    """
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1

    max_memory = f'{max_memory_mb}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # Handle distributed training
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    compute_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)

    logger.info(f'Loading base model {model_name_or_path}...')

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_quant_type=quant_type,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    if compute_dtype == torch.float16 and bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('Your GPU supports bfloat16, consider using --bf16 for better training')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype = compute_dtype

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    # Handle LLaMA-specific tokenizer configuration
    if 'llama' in model_name_or_path.lower() or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        logger.info('Adding special tokens for LLaMA tokenizer.')
        special_tokens = {}

        if model.config.eos_token_id is not None:
            special_tokens["eos_token"] = tokenizer.convert_ids_to_tokens(model.config.eos_token_id)

        if model.config.bos_token_id is not None:
            special_tokens["bos_token"] = tokenizer.convert_ids_to_tokens(model.config.bos_token_id)

        unk_id = (
            model.config.pad_token_id
            if model.config.pad_token_id not in (-1, None)
            else tokenizer.pad_token_id
        )
        if unk_id is not None:
            special_tokens["unk_token"] = tokenizer.convert_ids_to_tokens(unk_id)

        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # Load existing adapter or create new one
    if checkpoint_dir is not None:
        logger.info("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
    else:
        logger.info(f'Adding LoRA modules...')
        modules = find_all_linear_names(model, bits)
        logger.info(f'Target modules: {modules}')

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    # Ensure proper dtype for different module types
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    return model, tokenizer


def get_last_checkpoint(checkpoint_dir: str):
    """
    Find the last checkpoint in a training run.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        tuple: (checkpoint_path or None, is_completed)
    """
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed:
            return None, True  # Already finished

        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))

        if max_step == 0:
            return None, is_completed  # Training started, but no checkpoint

        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed

    return None, False  # First training
