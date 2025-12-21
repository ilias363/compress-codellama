"""
Wanda Pruning Pipeline for CodeLlama
Adapted from: https://github.com/locuslab/wanda

This script applies the Wanda pruning method to CodeLlama models.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import SEED, MODEL_CONFIG, PRUNING_CONFIG, setup_logging
from utils.io_utils import save_statistics
from lib.wanda.prune import prune_wanda, check_sparsity


logger = logging.getLogger(__name__)


def load_model(model_name: str, cache_dir: Path):
    """
    Load CodeLlama model.

    Args:
        model_name: Model name or path
        cache_dir: Cache directory for model weights

    Returns:
        Loaded model
    """
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto"
    )

    if hasattr(model.config, 'max_position_embeddings'):
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 2048  # default for CodeLlama

    logger.info(f"Model loaded. Sequence length: {model.seqlen}")
    return model


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
        default='./outputs/cached_llm_weights',
        help='Cache directory for model weights',
    )

    parser.add_argument(
        '--calib_dataset',
        type=str,
        default=PRUNING_CONFIG['calib_dataset'],
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
        default=PRUNING_CONFIG['output_dir'],
        help='Directory to save pruned model',
    )
    parser.add_argument(
        '--save_stats',
        action='store_true',
        default=PRUNING_CONFIG['save_stats'],
        help='Save pruning statistics',
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

    model = load_model(args.model, args.cache_dir)
    model.eval()

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

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

        logger.info("Pruning complete!")
    else:
        logger.warning("Sparsity ratio is 0, skipping pruning")

    logger.info("=" * 70)
    logger.info("Checking final sparsity...")
    final_sparsity = check_sparsity(model)
    logger.info(f"Final model sparsity: {final_sparsity:.4f}")
    logger.info("=" * 70)

    model_path = Path(args.output_dir) / "pruned_model"
    os.makedirs(model_path, exist_ok=True)
    logger.info(f"Saving pruned model to: {model_path}")

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    if args.save_stats:
        json_stats_path = os.path.join(args.output_dir, "pruning_stats.json")
        logger.info(f"Saving pruning statistics (JSON) to: {json_stats_path}")

        pruning_stats = {
            "model": args.model,
            "pruning_method": "wanda",
            "source": "https://github.com/locuslab/wanda",
            "sparsity_type": args.sparsity_type,
            "target_sparsity": float(args.sparsity_ratio),
            "actual_sparsity": float(final_sparsity),
            "calibration_samples": args.nsamples,
            "calibration_dataset": args.calib_dataset,
            "use_variant": args.use_variant,
            "random_seed": args.seed,
            "output_dir": args.output_dir,
        }

        save_statistics(pruning_stats, json_stats_path)

    logger.info("=" * 70)
    logger.info("Wanda pruning pipeline completed successfully!")
    logger.info(f"Pruned model saved to: {model_path}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
