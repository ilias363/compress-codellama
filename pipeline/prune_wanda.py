"""
Wanda Pruning Pipeline for CodeLlama
Adapted from: https://github.com/locuslab/wanda

This script applies the Wanda pruning method to CodeLlama models.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import SEED, MODEL_CONFIG, PATHS_CONFIG, DATASET_CONFIG, PRUNING_CONFIG, setup_logging, ensure_paths_exist
from utils.model_utils import load_model, load_tokenizer, get_model_size_mb, count_parameters
from utils.io_utils import save_json
from lib.wanda.prune import prune_wanda, check_sparsity


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Apply Wanda pruning to CodeLlama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model', type=str, default=MODEL_CONFIG['name'], help='CodeLlama model name or path'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=f"{PATHS_CONFIG["cache_dir"]}/llm_weights",
        help='Cache directory for model weights',
    )

    parser.add_argument(
        '--calib_dataset',
        type=str,
        default=f"{PATHS_CONFIG['datasets_dir']}/{DATASET_CONFIG['calib_file']}",
        help='Path to calibration dataset JSON file',
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=PRUNING_CONFIG['nsamples'],
        help='Number of calibration samples',
    )
    parser.add_argument(
        '--max_calib_seqlen',
        type=int,
        default=PRUNING_CONFIG['max_calib_seqlen'],
        help='Maximum sequence length for calibration (reduces memory usage)',
    )

    parser.add_argument(
        '--sparsity_ratio',
        type=float,
        default=PRUNING_CONFIG['sparsity_ratio'],
        help='Target sparsity ratio 0-1',
    )
    parser.add_argument(
        '--sparsity_type',
        type=str,
        default=PRUNING_CONFIG['sparsity_type'],
        choices=['unstructured', '4:8', '2:4'],
        help='Type of sparsity',
    )
    parser.add_argument(
        '--use_variant',
        action='store_true',
        default=PRUNING_CONFIG['use_variant'],
        help='Use Wanda variant with adaptive threshold',
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/pruned",
        help='Directory to save pruned model',
    )
    parser.add_argument(
        '--stats_dir',
        type=str,
        default=PATHS_CONFIG['stats_dir'],
        help='Directory to save pruning statistics',
    )

    parser.add_argument('--seed', type=int, default=SEED, help=f'Random seed')

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
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        if args.sparsity_ratio != 0.5:
            logger.warning(f"For structured {args.sparsity_type} sparsity, ratio must be 0.5. Adjusting...")
            args.sparsity_ratio = 0.5
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
        logger.info(f"Using structured {prune_n}:{prune_m} sparsity")
    else:
        logger.info(f"Using unstructured sparsity with ratio {args.sparsity_ratio}")

    model = load_model(args.model, cache_dir=args.cache_dir, set_seqlen=True)
    model.eval()

    tokenizer = load_tokenizer(args.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if hasattr(model, 'hf_device_map') and "lm_head" in model.hf_device_map:
        device = model.hf_device_map["lm_head"]
    logger.info(f"Using device: {device}")

    if not os.path.exists(args.calib_dataset):
        logger.error(f"Calibration dataset not found: {args.calib_dataset}")
        sys.exit(1)

    if args.sparsity_ratio > 0:
        logger.info("=" * 70)
        logger.info(f"Starting Wanda pruning")
        logger.info(f"  Calibration dataset: {args.calib_dataset}")
        logger.info(f"  Calibration samples: {args.nsamples}")
        logger.info(f"  Max calibration seqlen: {args.max_calib_seqlen}")
        logger.info(f"  Target sparsity: {args.sparsity_ratio}")
        logger.info(f"  Sparsity type: {args.sparsity_type}")
        logger.info(f"  Use variant: {args.use_variant}")
        logger.info("=" * 70)

        prune_start_time = time.time()

        prune_wanda(
            model=model,
            tokenizer=tokenizer,
            device=device,
            calib_dataset_path=args.calib_dataset,
            nsamples=args.nsamples,
            max_calib_seqlen=args.max_calib_seqlen,
            sparsity_ratio=args.sparsity_ratio,
            prune_n=prune_n,
            prune_m=prune_m,
            use_variant=args.use_variant,
        )

        pruning_time = time.time() - prune_start_time

        logger.info("Pruning complete!")
        logger.info(f"Pruning time: {pruning_time:.2f}s")
    else:
        logger.warning("Sparsity ratio is 0, skipping pruning")
        pruning_time = 0.0

    logger.info("=" * 70)
    logger.info("Checking final sparsity...")
    final_sparsity = check_sparsity(model)
    model_size_mb = get_model_size_mb(model)
    total_params = count_parameters(model)
    non_zero_params = int(total_params * (1 - final_sparsity))

    logger.info(f"Final model sparsity: {final_sparsity:.4f} ({final_sparsity*100:.2f}%)")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")
    logger.info("=" * 70)

    model_path = Path(args.output_dir)
    os.makedirs(model_path, exist_ok=True)
    logger.info(f"Saving pruned model to: {model_path}")

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    json_stats_path = Path(args.stats_dir) / f"pruning_stats_{int(time.time())}.json"
    logger.info(f"Saving pruning statistics (JSON) to: {json_stats_path}")

    pruning_stats = {
        "model": args.model,
        "model_size_mb": round(model_size_mb, 2),
        "total_parameters": total_params,
        "non_zero_parameters": non_zero_params,
        "pruning_method": "wanda",
        "sparsity_type": args.sparsity_type,
        "target_sparsity": float(args.sparsity_ratio),
        "target_sparsity_percent": round(args.sparsity_ratio * 100, 2),
        "actual_sparsity": float(final_sparsity),
        "actual_sparsity_percent": round(final_sparsity * 100, 2),
        "use_variant": args.use_variant,
        "calibration_dataset": args.calib_dataset,
        "calibration_samples": args.nsamples,
        "max_calibration_seqlen": args.max_calib_seqlen,
        "prune_n": prune_n,
        "prune_m": prune_m,
        "pruning_time_seconds": round(pruning_time, 2),
        "seed": args.seed,
        "output_model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
    }

    save_json(pruning_stats, json_stats_path)

    logger.info("=" * 70)
    logger.info("Wanda pruning pipeline completed successfully!")
    logger.info(f"Pruned model saved to: {model_path}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
