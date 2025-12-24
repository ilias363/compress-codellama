"""
Efficiency metrics evaluation script.

Measures model efficiency metrics:
- Inference speed (tokens/second, latency)
- Memory usage (GPU VRAM, peak memory)
- Model size (disk size, parameters, sparsity)
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
from utils.model_utils import load_model, load_tokenizer, check_sparsity, get_model_size_mb, count_parameters
from utils.io_utils import save_json
from lib.eval.efficiency import (
    get_gpu_memory_info,
    reset_memory_stats,
    measure_inference_latency,
    measure_throughput,
    get_model_disk_size,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model efficiency metrics (speed, memory, size)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_CONFIG['name'],
        help='Model path or HuggingFace model name',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=f"{PATHS_CONFIG['cache_dir']}/llm_weights",
        help='Cache directory for model weights',
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
        default=f"{PATHS_CONFIG['stats_dir']}/efficiency_results_{int(time.time())}.json",
        help='Path to save results (JSON)',
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace auth token for private/gated models',
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging except errors")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    ensure_paths_exist()

    torch.manual_seed(args.seed)

    logger.info(f"Loading model: {args.model}")

    reset_memory_stats()
    initial_gpu_memory = get_gpu_memory_info()

    load_start = time.time()
    model = load_model(args.model, cache_dir=args.cache_dir, set_seqlen=True, hf_token=args.hf_token)
    tokenizer = load_tokenizer(args.model, hf_token=args.hf_token)
    load_time = time.time() - load_start

    sparsity = check_sparsity(model)
    model_size_mb = get_model_size_mb(model)
    total_params = count_parameters(model)
    non_zero_params = int(total_params * (1 - sparsity))

    post_load_gpu_memory = get_gpu_memory_info()

    disk_size_info = get_model_disk_size(args.model)

    logger.info("=" * 70)
    logger.info("Model Information")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")
    logger.info(f"Sparsity: {sparsity*100:.2f}%")
    logger.info(f"Model size (memory): {model_size_mb:.2f} MB")
    logger.info(f"Model load time: {load_time:.2f}s")

    logger.info("=" * 70)
    logger.info("Measuring Inference Latency...")
    logger.info("=" * 70)

    latency_results = measure_inference_latency(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
    )

    logger.info(f"Average latency: {latency_results['avg_latency_seconds']:.4f}s")
    logger.info(f"Tokens per second: {latency_results['tokens_per_second']:.2f}")
    logger.info(f"Time per token: {latency_results['time_per_token_ms']:.2f}ms")

    throughput_results = {}
    if len(args.benchmark_batches) > 1 or args.benchmark_batches[0] > 1:
        logger.info("=" * 70)
        logger.info("Measuring Throughput at Different Batch Sizes...")
        logger.info("=" * 70)

        throughput_results = measure_throughput(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            batch_sizes=args.benchmark_batches,
            num_runs=max(3, args.num_runs // 2),
        )

        for batch_key, metrics in throughput_results.items():
            if "error" not in metrics:
                logger.info(
                    f"{batch_key}: {metrics['throughput_tokens_per_second']:.2f} tokens/s, peak memory: {metrics['peak_memory_gb']:.2f} GB"
                )

    final_gpu_memory = get_gpu_memory_info()

    results = {
        "model": args.model,
        "evaluation_type": "efficiency",
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "model_size": {
            "total_parameters": total_params,
            "non_zero_parameters": non_zero_params,
            "sparsity": round(sparsity, 4),
            "sparsity_percent": round(sparsity * 100, 2),
            "memory_size_mb": round(model_size_mb, 2),
            "memory_size_gb": round(model_size_mb / 1024, 2),
            **disk_size_info,
        },
        "inference_latency": latency_results,
        "throughput": throughput_results,
        "memory": {
            "model_load_time_seconds": round(load_time, 2),
            "initial_gpu_memory": initial_gpu_memory,
            "post_load_gpu_memory": post_load_gpu_memory,
            "final_gpu_memory": final_gpu_memory,
        },
        "settings": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "num_runs": args.num_runs,
            "num_warmup": args.num_warmup,
            "benchmark_batches": args.benchmark_batches,
        },
    }

    output_file = Path(args.output_file)
    save_json(results, output_file)

    logger.info("=" * 70)
    logger.info("Efficiency Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    logger.info(f"Sparsity: {sparsity*100:.2f}%")
    logger.info(f"Model size: {model_size_mb:.2f} MB ({model_size_mb/1024:.2f} GB)")
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
