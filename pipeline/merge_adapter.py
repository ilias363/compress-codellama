"""
Merge LoRA Adapter Pipeline
Merges QLoRA/LoRA adapters into the base model for efficient inference.

This script takes a base model and a PEFT adapter, merges them together,
and saves the resulting model which can be used without the PEFT library.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_utils import (
    SEED,
    PATHS_CONFIG,
    setup_logging,
    ensure_paths_exist,
)
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights into base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--base_model',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/pruned",
        help='Path to base model (pruned model)',
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/finetuned/adapter_model",
        help='Path to PEFT adapter directory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/merged",
        help='Directory to save merged model',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=f"{PATHS_CONFIG['cache_dir']}/llm_weights",
        help='Cache directory for model weights',
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default='float16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Torch dtype for the merged model',
    )
    parser.add_argument(
        '--save_safetensors',
        action='store_true',
        default=True,
        help='Save model in safetensors format (recommended)',
    )
    parser.add_argument(
        '--no_safetensors',
        action='store_true',
        default=False,
        help='Save model in pytorch bin format instead of safetensors',
    )

    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace auth token for accessing private/gated models',
    )
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging except errors")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    ensure_paths_exist()

    if not os.path.exists(args.adapter_path):
        logger.error(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    logger.info("=" * 70)
    logger.info("Merge LoRA Adapter Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Base model: {args.base_model}")
    logger.info(f"  Adapter path: {args.adapter_path}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Torch dtype: {args.torch_dtype}")
    logger.info("=" * 70)

    merge_start_time = time.time()

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        token=args.hf_token,
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        token=args.hf_token,
    )

    logger.info("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=torch_dtype,
    )

    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters (with adapter): {total_params_before:,}")
    logger.info(f"Trainable adapter parameters: {trainable_params:,}")

    logger.info("Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    total_params_after = sum(p.numel() for p in merged_model.parameters())
    logger.info(f"Total parameters (merged): {total_params_after:,}")

    os.makedirs(args.output_dir, exist_ok=True)

    use_safetensors = args.save_safetensors and not args.no_safetensors

    logger.info(f"Saving merged model to {args.output_dir}...")
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=use_safetensors,
    )

    logger.info("Saving tokenizer...")
    tokenizer.save_pretrained(args.output_dir)

    merge_time = time.time() - merge_start_time

    model_size_mb = sum(f.stat().st_size for f in Path(args.output_dir).glob("*.safetensors")) + sum(
        f.stat().st_size for f in Path(args.output_dir).glob("*.bin")
    )
    model_size_mb = model_size_mb / (1024 * 1024)

    logger.info("=" * 70)
    logger.info("Merge completed!")
    logger.info(f"  Time: {merge_time:.2f}s")
    logger.info(f"  Output size: {model_size_mb:.2f} MB")
    logger.info(f"  Saved to: {args.output_dir}")
    logger.info("=" * 70)

    stats = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "output_dir": args.output_dir,
        "torch_dtype": args.torch_dtype,
        "safetensors": use_safetensors,
        "total_params_with_adapter": total_params_before,
        "trainable_adapter_params": trainable_params,
        "total_params_merged": total_params_after,
        "model_size_mb": model_size_mb,
        "merge_time_seconds": merge_time,
        "timestamp": datetime.now().isoformat(),
    }

    stats_path = Path(PATHS_CONFIG['stats_dir']) / f"merge_stats_{int(time.time())}.json"
    save_json(stats, stats_path)
    logger.info(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
