import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.io_utils import load_dataset_file

logger = logging.getLogger(__name__)


def get_calibration_dataset(
    tokenizer,
    source: Union[str, Path, List[str]] = "HuggingFaceH4/ultrachat_200k",
    n_samples: int = 256,
    max_seq_length: int = 2048,
    split: str = "train_sft",
    seed: int = 42,
) -> Dataset:
    """
    Prepare calibration dataset for quantization.

    Args:
        tokenizer: Tokenizer to use for processing
        source: Dataset source - can be:
            - HuggingFace dataset ID (e.g., "HuggingFaceH4/ultrachat_200k")
            - Path to local JSON/JSONL file
            - List of text samples
        n_samples: Number of calibration samples
        max_seq_length: Maximum sequence length
        split: Dataset split (for HuggingFace datasets)
        seed: Random seed for shuffling

    Returns:
        Processed Dataset ready for quantization
    """
    source_path = Path(source) if isinstance(source, str) else None

    if isinstance(source, list):
        logger.info(f"Using {len(source)} provided text samples")
        ds = Dataset.from_dict({"text": source[:n_samples]})

    elif source_path and source_path.exists():
        logger.info(f"Loading calibration data from local file: {source}")
        data = load_dataset_file(source_path)

        texts = []
        for item in data[:n_samples]:
            if isinstance(item, dict):
                if "instruction" in item:
                    parts = [
                        item.get("instruction", ""),
                        item.get("input", ""),
                        item.get("output", ""),
                    ]
                    text = "\n".join(p for p in parts if p)
                elif "messages" in item:
                    messages = item["messages"]
                    text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
                else:
                    text = item.get("text") or item.get("content") or item.get("code") or str(item)
            else:
                text = str(item)
            texts.append(text.strip())

        ds = Dataset.from_dict({"text": texts})

    else:
        logger.info(f"Loading calibration data from HuggingFace: {source}")
        ds = load_dataset(source, split=f"{split}[:{n_samples}]")
        ds = ds.shuffle(seed=seed)

    # Preprocess for chat datasets (ultrachat format)
    def preprocess(example):
        if "messages" in example:
            return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
        elif "text" in example:
            return {"text": example["text"]}
        else:
            for col in ["content", "code", "instruction"]:
                if col in example:
                    return {"text": example[col]}
            return {"text": str(example)}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    logger.info(f"Prepared {len(ds)} calibration samples")
    return ds


def quantize_awq(
    model_path: str,
    output_path: str,
    calib_dataset: Union[str, Dataset] = "HuggingFaceH4/ultrachat_200k",
    n_samples: int = 256,
    max_seq_length: int = 2048,
    scheme: str = "W4A16",
    ignore_layers: Optional[List[str]] = None,
    duo_scaling: bool = True,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quantize model using AWQ (Activation-aware Weight Quantization).

    AWQ preserves important weights based on activation patterns,
    providing better quality than naive quantization.

    Args:
        model_path: Path to model or HuggingFace model ID
        output_path: Path to save quantized model
        calib_dataset: Calibration dataset (HF ID, path, or Dataset object)
        n_samples: Number of calibration samples
        max_seq_length: Maximum sequence length for calibration
        scheme: Quantization scheme (W4A16, W4A16_ASYM, W8A16, etc.)
        ignore_layers: Layers to skip (default: ["lm_head"])
        duo_scaling: Enable dual scaling for better accuracy
        trust_remote_code: Trust remote code for custom models
        torch_dtype: Data type for model loading ("auto", "float16", "bfloat16")
        hf_token: HuggingFace token for private models

    Returns:
        Dictionary with quantization statistics
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.utils import dispatch_for_generation

    if ignore_layers is None:
        ignore_layers = ["lm_head"]

    logger.info(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    if isinstance(calib_dataset, Dataset):
        ds = calib_dataset
    else:
        ds = get_calibration_dataset(
            tokenizer,
            source=calib_dataset,
            n_samples=n_samples,
            max_seq_length=max_seq_length,
        )

    recipe = [
        AWQModifier(
            ignore=ignore_layers,
            scheme=scheme,
            targets=["Linear"],
            duo_scaling=duo_scaling,
        ),
    ]

    logger.info(f"Applying AWQ quantization with scheme: {scheme}")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=n_samples,
    )

    logger.info("Testing quantized model generation...")
    dispatch_for_generation(model)
    input_ids = tokenizer("def hello_world():", return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=50)
    sample_output = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Sample output: {sample_output[:200]}...")

    logger.info(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    stats = {
        "method": "awq",
        "library": "llmcompressor",
        "model_path": str(model_path),
        "output_path": str(output_path),
        "scheme": scheme,
        "n_samples": n_samples,
        "max_seq_length": max_seq_length,
        "ignore_layers": ignore_layers,
        "duo_scaling": duo_scaling,
    }

    logger.info("AWQ quantization complete!")
    return stats


def quantize_gptq(
    model_path: str,
    output_path: str,
    calib_dataset: Union[str, Dataset] = "HuggingFaceH4/ultrachat_200k",
    n_samples: int = 512,
    max_seq_length: int = 2048,
    scheme: str = "W4A16",
    ignore_layers: Optional[List[str]] = None,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quantize model using GPTQ (Gradient-based Post-Training Quantization).

    GPTQ uses gradient information to minimize quantization error,
    often providing the best quality for weight-only quantization.

    Args:
        model_path: Path to model or HuggingFace model ID
        output_path: Path to save quantized model
        calib_dataset: Calibration dataset
        n_samples: Number of calibration samples (512 recommended)
        max_seq_length: Maximum sequence length
        scheme: Quantization scheme (W4A16, W8A8, etc.)
        ignore_layers: Layers to skip
        trust_remote_code: Trust remote code
        torch_dtype: Model data type
        hf_token: HuggingFace token

    Returns:
        Quantization statistics
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.utils import dispatch_for_generation

    if ignore_layers is None:
        ignore_layers = ["lm_head"]

    logger.info(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    if isinstance(calib_dataset, Dataset):
        ds = calib_dataset
    else:
        ds = get_calibration_dataset(
            tokenizer,
            source=calib_dataset,
            n_samples=n_samples,
            max_seq_length=max_seq_length,
        )

    recipe = GPTQModifier(
        targets="Linear",
        scheme=scheme,
        ignore=ignore_layers,
    )

    logger.info(f"Applying GPTQ quantization with scheme: {scheme}")

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=n_samples,
    )

    logger.info("Testing quantized model generation...")
    dispatch_for_generation(model)
    input_ids = tokenizer("def hello_world():", return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=50)
    sample_output = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Sample output: {sample_output[:200]}...")

    logger.info(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    stats = {
        "method": "gptq",
        "library": "llmcompressor",
        "model_path": str(model_path),
        "output_path": str(output_path),
        "scheme": scheme,
        "n_samples": n_samples,
        "max_seq_length": max_seq_length,
        "ignore_layers": ignore_layers,
    }

    logger.info("GPTQ quantization complete!")
    return stats


def quantize_fp8(
    model_path: str,
    output_path: str,
    scheme: str = "FP8_DYNAMIC",
    ignore_layers: Optional[List[str]] = None,
    trust_remote_code: bool = True,
    torch_dtype: str = "auto",
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quantize model to FP8 (8-bit floating point).

    FP8 quantization is fast (no calibration needed for dynamic mode)
    and provides good accuracy with 2x memory reduction.

    Args:
        model_path: Path to model
        output_path: Output path
        scheme: FP8 scheme (FP8_DYNAMIC, FP8_BLOCK)
        ignore_layers: Layers to skip
        trust_remote_code: Trust remote code
        torch_dtype: Model data type
        hf_token: HuggingFace token

    Returns:
        Quantization statistics
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.utils import dispatch_for_generation

    if ignore_layers is None:
        ignore_layers = ["lm_head"]

    logger.info(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=ignore_layers,
    )

    logger.info(f"Applying FP8 quantization with scheme: {scheme}")

    oneshot(model=model, recipe=recipe)

    logger.info("Testing quantized model generation...")
    dispatch_for_generation(model)
    input_ids = tokenizer("def hello_world():", return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=50)
    sample_output = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Sample output: {sample_output[:200]}...")

    logger.info(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    stats = {
        "method": "fp8",
        "library": "llmcompressor",
        "model_path": str(model_path),
        "output_path": str(output_path),
        "scheme": scheme,
        "ignore_layers": ignore_layers,
    }

    logger.info("FP8 quantization complete!")
    return stats


def load_quantized_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    hf_token: Optional[str] = None,
):
    """
    Load a model quantized with llm-compressor.

    Models saved in compressed-tensors format can be loaded directly
    with standard transformers AutoModelForCausalLM.

    Args:
        model_path: Path to quantized model
        device_map: Device placement strategy
        torch_dtype: Data type for computation
        trust_remote_code: Trust remote code
        hf_token: HuggingFace token

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading quantized model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )

    logger.info("Model loaded successfully!")
    return model, tokenizer
