"""
Efficiency metrics evaluation script using vLLM.

This script is optimized for quantized models (AWQ, GPTQ, compressed-tensors format)
as vLLM has native kernel support for these formats.

Measures model efficiency metrics:
- Inference speed (tokens/second, latency)
- Memory usage (GPU VRAM)
- Throughput at different batch sizes
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_utils import (
    PATHS_CONFIG,
    SEED,
    MODEL_CONFIG,
    EFFICIENCY_EVAL_CONFIG,
    setup_logging,
    ensure_paths_exist,
)
from utils.io_utils import save_json
from lib.eval.efficiency import (
    get_gpu_memory_info,
    measure_vllm_latency,
    measure_vllm_throughput,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model efficiency metrics using vLLM (optimized for quantized models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_CONFIG['name'],
        help='Model path or HuggingFace model name',
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=EFFICIENCY_EVAL_CONFIG['max_new_tokens'],
        help='Number of tokens to generate in benchmarks',
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=EFFICIENCY_EVAL_CONFIG['num_runs'],
        help='Number of timed runs for latency measurement',
    )
    parser.add_argument(
        '--num_warmup',
        type=int,
        default=EFFICIENCY_EVAL_CONFIG['num_warmup'],
        help='Number of warmup runs before timing',
    )
    parser.add_argument(
        '--benchmark_batches',
        type=int,
        nargs='+',
        default=EFFICIENCY_EVAL_CONFIG['benchmark_batches'],
        help='Batch sizes to benchmark for throughput',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=EFFICIENCY_EVAL_CONFIG['prompt'],
        help='Prompt to use for generation benchmarks',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save results (JSON). Auto-generated if not specified.',
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism (use 2 for dual T4 setup)',
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization for vLLM (0.0 to 1.0)',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16'],
        help='Data type for model weights',
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging except errors")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    ensure_paths_exist()

    torch.manual_seed(args.seed)

    # Set output file path
    if args.output_file is None:
        output_dir = Path(PATHS_CONFIG['stats_dir']) / "efficiency-eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_file = str(output_dir / f"efficiency_vllm_{int(time.time())}.json")

    logger.info("=" * 70)
    logger.info("vLLM Efficiency Evaluation")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")

    # Import vLLM
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
        sys.exit(1)

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    initial_gpu_memory = get_gpu_memory_info()

    # Load model with vLLM
    logger.info("Loading model with vLLM...")
    load_start = time.time()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.2f}s")

    post_load_gpu_memory = get_gpu_memory_info()

    # Create sampling params
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,  # Deterministic for benchmarking
    )

    # Measure latency
    logger.info("=" * 70)
    logger.info("Measuring Inference Latency...")
    logger.info("=" * 70)

    latency_results = measure_vllm_latency(
        llm=llm,
        sampling_params=sampling_params,
        prompt=args.prompt,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    logger.info(f"Average latency: {latency_results['avg_latency_seconds']:.4f}s")
    logger.info(f"Tokens per second: {latency_results['tokens_per_second']:.2f}")
    logger.info(f"Time per token: {latency_results['time_per_token_ms']:.2f}ms")

    # Measure throughput
    logger.info("=" * 70)
    logger.info("Measuring Throughput at Different Batch Sizes...")
    logger.info("=" * 70)

    throughput_results = measure_vllm_throughput(
        llm=llm,
        sampling_params=sampling_params,
        prompt=args.prompt,
        batch_sizes=args.benchmark_batches,
        num_runs=max(3, args.num_runs // 2),
    )

    final_gpu_memory = get_gpu_memory_info()

    # Compile results
    results = {
        "model": args.model,
        "evaluation_type": "efficiency_vllm",
        "inference_engine": "vllm",
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "inference_latency": latency_results,
        "throughput": throughput_results,
        "memory": {
            "model_load_time_seconds": round(load_time, 2),
            "initial_gpu_memory": initial_gpu_memory,
            "post_load_gpu_memory": post_load_gpu_memory,
            "final_gpu_memory": final_gpu_memory,
        },
        "vllm_settings": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
        },
        "settings": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "num_runs": args.num_runs,
            "num_warmup": args.num_warmup,
            "benchmark_batches": args.benchmark_batches,
        },
    }

    # Save results
    output_file = Path(args.output_file)
    save_json(results, output_file)
    logger.info(f"Results saved to: {output_file}")

    # Summary
    logger.info("=" * 70)
    logger.info("vLLM Efficiency Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Inference Engine: vLLM")
    logger.info(f"Inference speed: {latency_results['tokens_per_second']:.2f} tokens/s")
    logger.info(
        f"Latency: {latency_results['avg_latency_seconds']*1000:.1f}ms for {latency_results['tokens_generated']:.0f} tokens"
    )
    if post_load_gpu_memory.get("gpu_available"):
        logger.info(f"GPU memory (loaded): {post_load_gpu_memory.get('gpu_0_allocated_gb', 'N/A')} GB")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
