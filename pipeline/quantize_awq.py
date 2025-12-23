"""
AWQ Quantization Pipeline for CodeLlama
Using AutoAWQ: https://github.com/casper-hansen/AutoAWQ

This script applies AWQ (Activation-aware Weight Quantization) to CodeLlama models
using AutoAWQ which provides a high-level API and pre-compiled CUDA kernels.
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
    SEED,
    PATHS_CONFIG,
    DATASET_CONFIG,
    QUANTIZATION_CONFIG,
    setup_logging,
    ensure_paths_exist,
)
from utils.io_utils import save_json
from lib.awq.quantize import quantize_model


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Apply AWQ quantization to CodeLlama models using AutoAWQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/finetuned",
        help='Model path (typically the finetuned model after pruning recovery)',
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace auth token for accessing private/gated models',
    )

    parser.add_argument(
        '--calib_data',
        type=str,
        default=f"{PATHS_CONFIG['datasets_dir']}/{DATASET_CONFIG['awq_calib_file']}",
        help='Calibration data: local file path or "pileval" for default AWQ calibration',
    )
    parser.add_argument(
        '--max_calib_samples',
        type=int,
        default=QUANTIZATION_CONFIG['n_samples'],
        help='Maximum number of calibration samples (128 is standard for AWQ)',
    )
    parser.add_argument(
        '--max_calib_seq_len',
        type=int,
        default=QUANTIZATION_CONFIG['seqlen'],
        help='Maximum sequence length for calibration samples',
    )
    parser.add_argument(
        '--n_parallel_calib_samples',
        type=int,
        default=QUANTIZATION_CONFIG['n_parallel_calib_samples'],
        help='Number of parallel calibration samples (lower = less memory).',
    )

    parser.add_argument(
        '--w_bit',
        type=int,
        default=QUANTIZATION_CONFIG['w_bit'],
        choices=[2, 3, 4, 8],
        help='Weight quantization bit width (4-bit is standard for AWQ)',
    )
    parser.add_argument(
        '--q_group_size',
        type=int,
        default=QUANTIZATION_CONFIG['q_group_size'],
        help='Group size for group-wise quantization (128 is standard)',
    )
    parser.add_argument(
        '--zero_point',
        action='store_true',
        default=QUANTIZATION_CONFIG['zero_point'],
        help='Enable zero-point quantization',
    )
    parser.add_argument(
        '--version',
        type=str,
        default=QUANTIZATION_CONFIG['version'],
        choices=['GEMM', 'GEMV'],
        help='AWQ kernel version: GEMM (batch), GEMV (single)',
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/quantized",
        help='Directory to save quantized model',
    )
    parser.add_argument(
        '--stats_dir',
        type=str,
        default=PATHS_CONFIG['stats_dir'],
        help='Directory to save quantization statistics',
    )

    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all logging except errors",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    ensure_paths_exist()

    logger.info(f"Setting random seed: {args.seed}")
    torch.manual_seed(args.seed)

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    logger.info("=" * 70)
    logger.info("AWQ Quantization Configuration (AutoAWQ)")
    logger.info("=" * 70)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Calibration data: {args.calib_data}")
    logger.info(f"  Max calibration samples: {args.max_calib_samples}")
    logger.info(f"  Max calibration seq len: {args.max_calib_seq_len}")
    logger.info(f"  Parallel calibration samples: {args.n_parallel_calib_samples}")
    logger.info(f"  Weight bits: {args.w_bit}")
    logger.info(f"  Group size: {args.q_group_size}")
    logger.info(f"  Zero-point: {quant_config['zero_point']}")
    logger.info(f"  Kernel version: {quant_config['version']}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("=" * 70)

    calib_data = args.calib_data
    calib_path = Path(args.calib_data)

    if not calib_path.exists() and args.calib_data != "pileval":
        logger.warning(f"Calibration data not found: {args.calib_data}")
        logger.info("Falling back to 'pileval' calibration dataset")
        calib_data = "pileval"

    logger.info("Starting AWQ quantization...")
    start_time = time.time()

    try:
        stats = quantize_model(
            model_path=args.model,
            output_path=args.output_dir,
            quant_config=quant_config,
            calib_data=calib_data,
            max_calib_samples=args.max_calib_samples,
            max_calib_seq_len=args.max_calib_seq_len,
            n_parallel_calib_samples=args.n_parallel_calib_samples,
            hf_token=args.hf_token,
        )
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise

    total_time = time.time() - start_time

    logger.info("=" * 70)
    logger.info("Quantization Results")
    logger.info("=" * 70)
    logger.info(f"Total quantization time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Quantized model saved to: {args.output_dir}")
    logger.info("=" * 70)

    stats_path = Path(args.stats_dir) / f"quantization_stats_{int(time.time())}.json"
    logger.info(f"Saving quantization statistics to: {stats_path}")

    quantization_stats = {
        "method": "awq",
        "library": "autoawq",
        "input_model": args.model,
        "output_model": args.output_dir,
        "quant_config": quant_config,
        "calibration": {
            "data": str(calib_data),
            "max_samples": args.max_calib_samples,
            "max_seq_len": args.max_calib_seq_len,
            "n_parallel_samples": args.n_parallel_calib_samples,
        },
        "total_time_seconds": round(total_time, 2),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    save_json(quantization_stats, stats_path)

    logger.info("=" * 70)
    logger.info("AWQ quantization pipeline completed successfully!")
    logger.info(f"Quantized model saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
