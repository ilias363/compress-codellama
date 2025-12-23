"""
AWQ Quantization using AutoAWQ
Based on: https://github.com/casper-hansen/AutoAWQ
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.io_utils import load_dataset_file

logger = logging.getLogger(__name__)

# Default quantization config
DEFAULT_QUANT_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}


def get_calib_data(file_path: Union[str, Path], n_samples: int = 128) -> List[str]:
    """
    Load calibration data from a local JSON/JSONL file.

    Args:
        file_path: Path to the calibration data file
        n_samples: Number of samples to use

    Returns:
        List of text samples for calibration
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {file_path}")

    data = load_dataset_file(file_path)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of samples, got {type(data)}")

    samples = []
    for item in data[:n_samples]:
        if isinstance(item, dict):
            # Handle Pipeline/prepare_dataset.py format: instruction + input + output
            if "instruction" in item:
                parts = [
                    item.get("instruction", ""),
                    item.get("input", ""),
                    item.get("output", ""),
                ]
                text = "\n".join(p for p in parts if p)
            else:
                # Fallback to common keys
                text = item.get("text") or item.get("content") or item.get("code") or ""
        elif isinstance(item, str):
            text = item
        else:
            continue

        samples.append(text.strip())

    logger.info(f"Loaded {len(samples)} calibration samples from {file_path}")
    return samples


def quantize_model(
    model_path: str,
    output_path: str,
    quant_config: Optional[Dict[str, Any]] = None,
    calib_data: Union[str, List[str]] = "pileval",
    max_calib_samples: int = 128,
    max_calib_seq_len: int = 512,
    n_parallel_calib_samples: Optional[int] = None,
    device_map: str = None,
    trust_remote_code: bool = True,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quantize a model using AutoAWQ.

    Args:
        model_path: Path to the model to quantize (HuggingFace or local path)
        output_path: Path to save the quantized model
        quant_config: Quantization configuration. Defaults to:
            - zero_point: True (asymmetric quantization)
            - q_group_size: 128 (group size for quantization)
            - w_bit: 4 (4-bit quantization)
            - version: "GEMM" (optimized inference kernel)
        calib_data: Calibration data - either "pileval" (default), a dataset name,
                   a path to local file, or a list of text samples
        max_calib_samples: Maximum number of calibration samples
        max_calib_seq_len: Maximum sequence length for calibration
        n_parallel_calib_samples: Number of parallel samples during calibration.
                                  Lower values use less memory. None = all at once.
        device_map: Device map for loading the model
        trust_remote_code: Whether to trust remote code in the model
        hf_token: HuggingFace token for private models

    Returns:
        Dictionary with quantization statistics
    """
    if quant_config is None:
        quant_config = DEFAULT_QUANT_CONFIG.copy()
    else:
        config = DEFAULT_QUANT_CONFIG.copy()
        config.update(quant_config)
        quant_config = config

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Quantization config: {quant_config}")

    # Load model - use safetensors=True for sharded safetensors models
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        use_cache=False,
        token=hf_token,
        safetensors=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    if isinstance(calib_data, str) and Path(calib_data).exists():
        logger.info(f"Loading calibration data from local file: {calib_data}")
        calib_samples = get_calib_data(calib_data, n_samples=max_calib_samples)
    elif isinstance(calib_data, list):
        calib_samples = calib_data[:max_calib_samples]
    else:
        # Dataset name (e.g., "pileval") - AutoAWQ handles this
        calib_samples = calib_data

    logger.info("Starting quantization...")

    if n_parallel_calib_samples is None:
        # Auto-adjust based on available memory
        n_parallel_calib_samples = 32

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_samples,
        max_calib_samples=max_calib_samples,
        max_calib_seq_len=max_calib_seq_len,
        n_parallel_calib_samples=n_parallel_calib_samples,
    )

    logger.info(f"Saving quantized model to: {output_path}")

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    stats = {
        "model_path": str(model_path),
        "output_path": str(output_path),
        "quant_config": quant_config,
        "max_calib_samples": max_calib_samples,
        "max_calib_seq_len": max_calib_seq_len,
        "n_parallel_calib_samples": n_parallel_calib_samples,
    }

    logger.info("Quantization complete!")
    return stats


def load_quantized_model(
    model_path: str,
    max_seq_len: int = 2048,
    fuse_layers: bool = True,
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """
    Load a quantized AWQ model for inference.

    Args:
        model_path: Path to the quantized model
        max_seq_len: Maximum sequence length
        fuse_layers: Whether to fuse layers for faster inference
        device_map: Device map for loading
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading quantized model from: {model_path}")

    model = AutoAWQForCausalLM.from_quantized(
        model_path,
        max_seq_len=max_seq_len,
        fuse_layers=fuse_layers,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    return model, tokenizer
