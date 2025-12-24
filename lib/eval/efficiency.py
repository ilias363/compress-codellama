"""
Efficiency metrics evaluation.

Measures model efficiency:
- Inference latency and speed
- Memory usage
- Model size and sparsity
- Throughput benchmarks
"""

import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with GPU memory information
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False}

    info = {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    }

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB

        info[f"gpu_{i}_allocated_gb"] = round(allocated, 2)
        info[f"gpu_{i}_reserved_gb"] = round(reserved, 2)
        info[f"gpu_{i}_max_allocated_gb"] = round(max_allocated, 2)
        info[f"gpu_{i}_total_gb"] = round(total, 2)
        info[f"gpu_{i}_free_gb"] = round(total - allocated, 2)

    return info


def reset_memory_stats():
    """Reset GPU memory statistics and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


@torch.no_grad()
def measure_inference_latency(
    model,
    tokenizer,
    prompt: str = "def hello_world():",
    max_new_tokens: int = 50,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Measure inference latency for text generation.

    Args:
        model: The model to benchmark
        tokenizer: The tokenizer
        prompt: Input prompt for generation
        max_new_tokens: Number of tokens to generate
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        device: Device to use

    Returns:
        Dictionary with latency metrics
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    # Warmup runs
    logger.info(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    logger.info(f"Running {num_runs} timed iterations...")
    latencies = []
    tokens_generated = []

    reset_memory_stats()

    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        latency = end_time - start_time
        num_tokens = outputs.shape[1] - input_length

        latencies.append(latency)
        tokens_generated.append(num_tokens)

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    min_latency = min(latencies)
    max_latency = max(latencies)

    # Calculate tokens per second
    tokens_per_second = avg_tokens / avg_latency if avg_latency > 0 else 0
    time_per_token_ms = (avg_latency / avg_tokens * 1000) if avg_tokens > 0 else 0

    return {
        "avg_latency_seconds": round(avg_latency, 4),
        "min_latency_seconds": round(min_latency, 4),
        "max_latency_seconds": round(max_latency, 4),
        "tokens_generated": round(avg_tokens, 1),
        "tokens_per_second": round(tokens_per_second, 2),
        "time_per_token_ms": round(time_per_token_ms, 2),
        "num_runs": num_runs,
        "max_new_tokens_setting": max_new_tokens,
        "input_length": input_length,
    }


@torch.no_grad()
def measure_throughput(
    model,
    tokenizer,
    prompt: str = "def hello_world():",
    max_new_tokens: int = 50,
    batch_sizes: List[int] = [1, 2, 4, 8],
    num_runs: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Measure throughput at different batch sizes.

    Args:
        model: The model to benchmark
        tokenizer: The tokenizer
        prompt: Input prompt for generation
        max_new_tokens: Number of tokens to generate
        batch_sizes: List of batch sizes to test
        num_runs: Number of runs per batch size
        device: Device to use

    Returns:
        Dictionary with throughput metrics for each batch size
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    results = {}

    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")

        # Create batched input
        inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True).to(device)

        # Warmup
        try:
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch_size={batch_size}, skipping")
                results[f"batch_{batch_size}"] = {"error": "OOM"}
                torch.cuda.empty_cache()
                continue
            raise

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        latencies = []
        reset_memory_stats()

        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

        avg_latency = sum(latencies) / len(latencies)
        total_tokens = batch_size * max_new_tokens
        throughput = total_tokens / avg_latency

        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "avg_latency_seconds": round(avg_latency, 4),
            "throughput_tokens_per_second": round(throughput, 2),
            "peak_memory_gb": round(peak_memory, 2),
        }

        torch.cuda.empty_cache()

    return results


def get_model_disk_size(model_path: str) -> Dict[str, float]:
    """Get model size on disk."""
    path = Path(model_path)

    if not path.exists():
        return {"disk_size_gb": None, "note": "Model loaded from HuggingFace Hub"}

    total_size = 0
    file_count = 0

    if path.is_file():
        total_size = path.stat().st_size
        file_count = 1
    else:
        for f in path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
                file_count += 1

    return {
        "disk_size_gb": round(total_size / 1024**3, 2),
        "disk_size_mb": round(total_size / 1024**2, 2),
        "file_count": file_count,
    }


# ============================================================================
# vLLM-based efficiency functions (for quantized models with optimized kernels)
# ============================================================================


def measure_vllm_latency(
    llm,
    sampling_params,
    prompt: str,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, float]:
    """
    Measure inference latency with vLLM.

    Args:
        llm: vLLM LLM instance
        sampling_params: vLLM SamplingParams instance
        prompt: Input prompt for generation
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs

    Returns:
        Dictionary with latency metrics
    """
    # Warmup runs
    logger.info(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = llm.generate([prompt], sampling_params)

    # Timed runs
    latencies = []
    total_tokens = []

    logger.info(f"Running {num_runs} timed iterations...")
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        tokens_generated = len(outputs[0].outputs[0].token_ids)
        latencies.append(elapsed)
        total_tokens.append(tokens_generated)

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(total_tokens) / len(total_tokens)
    tokens_per_second = avg_tokens / avg_latency if avg_latency > 0 else 0

    return {
        "avg_latency_seconds": round(avg_latency, 4),
        "min_latency_seconds": round(min(latencies), 4),
        "max_latency_seconds": round(max(latencies), 4),
        "tokens_generated": avg_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "time_per_token_ms": round((avg_latency / avg_tokens) * 1000, 2) if avg_tokens > 0 else 0,
        "num_runs": num_runs,
        "max_new_tokens_setting": sampling_params.max_tokens,
    }


def measure_vllm_throughput(
    llm,
    sampling_params,
    prompt: str,
    batch_sizes: List[int],
    num_runs: int = 5,
) -> Dict[str, Any]:
    """
    Measure throughput at different batch sizes with vLLM.

    Args:
        llm: vLLM LLM instance
        sampling_params: vLLM SamplingParams instance
        prompt: Input prompt for generation
        batch_sizes: List of batch sizes to test
        num_runs: Number of runs per batch size

    Returns:
        Dictionary with throughput metrics for each batch size
    """
    results = {}

    for batch_size in batch_sizes:
        try:
            prompts = [prompt] * batch_size
            latencies = []
            total_tokens = []

            # Warmup
            _ = llm.generate(prompts, sampling_params)

            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling_params)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start

                tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                latencies.append(elapsed)
                total_tokens.append(tokens)

            avg_latency = sum(latencies) / len(latencies)
            avg_tokens = sum(total_tokens) / len(total_tokens)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "avg_latency_seconds": round(avg_latency, 4),
                "throughput_tokens_per_second": round(avg_tokens / avg_latency, 2) if avg_latency > 0 else 0,
                "peak_memory_gb": round(peak_memory, 2),
            }

            logger.info(
                f"Batch {batch_size}: {results[f'batch_{batch_size}']['throughput_tokens_per_second']:.2f} tokens/s"
            )

        except Exception as e:
            logger.warning(f"Batch size {batch_size} failed: {e}")
            results[f"batch_{batch_size}"] = {"batch_size": batch_size, "error": str(e)}

    return results
