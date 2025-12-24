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
from lib.llmcompressor.quantize import quantize_awq, quantize_gptq, quantize_fp8


logger = logging.getLogger(__name__)


# Quantization schemes available
SCHEMES = {
    "awq": ["W4A16", "W4A16_ASYM", "W8A16"],
    "gptq": ["W4A16", "W8A8", "W8A16"],
    "fp8": ["FP8_DYNAMIC", "FP8_BLOCK"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Quantize models using LLM Compressor (vLLM project)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/merged",
        help="Model path (local or HuggingFace model ID)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace auth token for private/gated models",
    )

    # Quantization method
    parser.add_argument(
        "--method",
        type=str,
        default=QUANTIZATION_CONFIG["method"],
        choices=["awq", "gptq", "fp8"],
        help="Quantization method: awq (recommended), gptq, or fp8",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=QUANTIZATION_CONFIG["scheme"],
        help="Quantization scheme (default: W4A16 for awq/gptq, FP8_DYNAMIC for fp8)",
    )

    # Calibration arguments
    parser.add_argument(
        "--calib_data",
        type=str,
        default=f"{PATHS_CONFIG['datasets_dir']}/{DATASET_CONFIG['awq_calib_file']}",
        help="Calibration data: local file path or HuggingFace dataset ID",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=QUANTIZATION_CONFIG["n_samples"],
        help="Number of calibration samples (256 for AWQ, 512 for GPTQ recommended)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=QUANTIZATION_CONFIG["max_seq_length"],
        help="Maximum sequence length for calibration",
    )

    # AWQ specific
    parser.add_argument(
        "--duo_scaling",
        action="store_true",
        default=QUANTIZATION_CONFIG["duo_scaling"],
        help="Enable dual scaling for AWQ (better accuracy)",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/quantized",
        help="Directory to save quantized model",
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        default=PATHS_CONFIG["stats_dir"],
        help="Directory to save quantization statistics",
    )

    # General
    parser.add_argument(
        "--ignore_layers",
        type=str,
        nargs="+",
        default=QUANTIZATION_CONFIG["ignore_layers"],
        help="Layers to skip during quantization",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Data type for model loading",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress logging except errors",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    ensure_paths_exist()

    logger.info(f"Setting random seed: {args.seed}")
    torch.manual_seed(args.seed)

    # Set default scheme based on method if FP8
    if args.method == "fp8" and args.scheme == "W4A16":
        args.scheme = "FP8_DYNAMIC"

    if args.scheme not in SCHEMES[args.method]:
        logger.warning(
            f"Scheme {args.scheme} may not be supported for {args.method}. "
            f"Available: {SCHEMES[args.method]}"
        )

    calib_data = args.calib_data
    calib_path = Path(args.calib_data)

    if not calib_path.exists():
        if args.method == "fp8":
            logger.info("FP8 quantization doesn't require calibration data")
            calib_data = None
        else:
            logger.info(f"Local calibration data not found: {args.calib_data}")
            logger.info("Using HuggingFaceH4/ultrachat_200k for calibration")
            calib_data = "HuggingFaceH4/ultrachat_200k"

    logger.info("=" * 70)
    logger.info("LLM Compressor Quantization Configuration")
    logger.info("=" * 70)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Scheme: {args.scheme}")
    if args.method != "fp8":
        logger.info(f"  Calibration data: {calib_data}")
        logger.info(f"  Calibration samples: {args.n_samples}")
        logger.info(f"  Max sequence length: {args.max_seq_length}")
    if args.method == "awq":
        logger.info(f"  Duo scaling: {args.duo_scaling}")
    logger.info(f"  Ignore layers: {args.ignore_layers}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("=" * 70)

    logger.info(f"Starting {args.method.upper()} quantization...")
    start_time = time.time()

    try:
        if args.method == "awq":
            stats = quantize_awq(
                model_path=args.model,
                output_path=args.output_dir,
                calib_dataset=calib_data,
                n_samples=args.n_samples,
                max_seq_length=args.max_seq_length,
                scheme=args.scheme,
                ignore_layers=args.ignore_layers,
                duo_scaling=args.duo_scaling,
                torch_dtype=args.torch_dtype,
                hf_token=args.hf_token,
            )

        elif args.method == "gptq":
            stats = quantize_gptq(
                model_path=args.model,
                output_path=args.output_dir,
                calib_dataset=calib_data,
                n_samples=args.n_samples,
                max_seq_length=args.max_seq_length,
                scheme=args.scheme,
                ignore_layers=args.ignore_layers,
                torch_dtype=args.torch_dtype,
                hf_token=args.hf_token,
            )

        elif args.method == "fp8":
            stats = quantize_fp8(
                model_path=args.model,
                output_path=args.output_dir,
                scheme=args.scheme,
                ignore_layers=args.ignore_layers,
                torch_dtype=args.torch_dtype,
                hf_token=args.hf_token,
            )

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise

    total_time = time.time() - start_time

    logger.info("=" * 70)
    logger.info("Quantization Results")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Quantized model saved to: {args.output_dir}")
    logger.info("=" * 70)

    stats_path = Path(args.stats_dir) / f"quantization_{args.method}_{int(time.time())}.json"
    logger.info(f"Saving statistics to: {stats_path}")

    full_stats = {
        **stats,
        "total_time_seconds": round(total_time, 2),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    save_json(full_stats, stats_path)

    logger.info("=" * 70)
    logger.info(f"{args.method.upper()} quantization completed successfully!")
    logger.info("")
    logger.info("The quantized model is saved in compressed-tensors format.")
    logger.info("It can be loaded with standard transformers:")
    logger.info("")
    logger.info("  from transformers import AutoModelForCausalLM")
    logger.info(f'  model = AutoModelForCausalLM.from_pretrained("{args.output_dir}")')
    logger.info("")
    logger.info("Or used directly with vLLM for fast inference.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
