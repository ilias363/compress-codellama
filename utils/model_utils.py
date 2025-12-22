import logging
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """Load and configure a tokenizer."""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    return tokenizer


def load_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    cache_dir: str = None,
    low_cpu_mem_usage: bool = True,
    set_seqlen: bool = False,
) -> PreTrainedModel:
    """
    Load a causal language model.

    Args:
        model_name: Model name or path
        device_map: Device map for model placement
        torch_dtype: Data type for model weights
        trust_remote_code: Trust remote code from HuggingFace
        load_in_8bit: Load model in 8-bit quantization
        load_in_4bit: Load model in 4-bit quantization
        cache_dir: Cache directory for model weights
        low_cpu_mem_usage: Use low CPU memory mode
        set_seqlen: If True, set model.seqlen attribute based on config

    Returns:
        Loaded model
    """
    logger.info(f"Loading model: {model_name}")

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit quantization with BitsAndBytes")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization with BitsAndBytes")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        cache_dir=cache_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    # Set seqlen attribute for pruning compatibility
    if set_seqlen:
        if hasattr(model.config, 'max_position_embeddings'):
            model.seqlen = model.config.max_position_embeddings
        else:
            model.seqlen = 2048  # default for CodeLlama
        logger.info(f"Model seqlen set to: {model.seqlen}")

    logger.info(
        f"Model loaded on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}"
    )
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    return total_size_mb


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def check_sparsity(model: nn.Module) -> float:
    """Calculate the overall sparsity of the model (percentage of zero weights)."""
    total_zeros = 0
    total_params = 0

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            zeros = (param == 0).sum().item()
            total = param.numel()
            total_zeros += zeros
            total_params += total

    return total_zeros / total_params if total_params > 0 else 0.0


def get_trainable_parameters(model: nn.Module, bits: int = 32) -> dict:
    """
    Get detailed information about trainable parameters in the model.
    
    Args:
        model: The model to analyze
        bits: Quantization bits (4, 8, 16, or 32). For 4-bit quantized models,
              the trainable param count is divided by 2 as they use half the
              storage of 8-bit parameters.

    Returns:
        Dict with trainable_params, all_params, trainable_percent
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # For 4-bit quantized models, trainable params are effectively half
    # because 4-bit uses half the storage of 8-bit
    if bits == 4:
        trainable_params_display = trainable_params / 2
    else:
        trainable_params_display = trainable_params

    trainable_pct = 100 * trainable_params / all_params if all_params > 0 else 0

    return {
        "trainable_params": int(trainable_params_display),
        "trainable_params_raw": trainable_params,
        "all_params": all_params,
        "trainable_percent": trainable_pct,
    }


def get_dtype_distribution(model: nn.Module) -> dict:
    """
    Get distribution of parameter data types in the model.

    Returns:
        Dict mapping dtype to count and percentage
    """
    dtypes = {}
    total = 0

    for _, p in model.named_parameters():
        dtype = str(p.dtype)
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
        total += p.numel()

    result = {}
    for dtype, count in dtypes.items():
        result[dtype] = {"count": count, "percentage": 100 * count / total if total > 0 else 0}

    return result
