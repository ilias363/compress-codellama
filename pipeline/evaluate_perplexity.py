"""
Perplexity evaluation script.

Evaluates language model perplexity on WikiText-2 dataset.
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
    PERPLEXITY_EVAL_CONFIG,
    setup_logging,
    ensure_paths_exist,
)
from utils.model_utils import load_model, load_tokenizer, check_sparsity, get_model_size_mb, count_parameters
from utils.io_utils import save_json
from lib.eval.perplexity import evaluate_perplexity, get_wikitext2

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model perplexity on WikiText-2",
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
        '--seqlen',
        type=int,
        default=PERPLEXITY_EVAL_CONFIG['seqlen'],
        help='Sequence length for evaluation',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=PERPLEXITY_EVAL_CONFIG['batch_size'],
        help='Batch size for evaluation',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=f"{PATHS_CONFIG['stats_dir']}/perplexity_results_{int(time.time())}.json",
        help='Path to save results (JSON).',
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace auth token for private/gated models',
    )
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging except errors")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    ensure_paths_exist()

    torch.manual_seed(args.seed)

    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, cache_dir=args.cache_dir, set_seqlen=True, hf_token=args.hf_token)
    tokenizer = load_tokenizer(args.model, hf_token=args.hf_token)

    sparsity = check_sparsity(model)
    model_size_mb = get_model_size_mb(model)
    total_params = count_parameters(model)
    non_zero_params = int(total_params * (1 - sparsity))

    logger.info(f"Model sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    logger.info(f"Total parameters: {total_params:,}")

    logger.info("=" * 70)
    logger.info("Evaluating Perplexity on WikiText-2...")
    logger.info("=" * 70)

    wikitext2_data = get_wikitext2(tokenizer, seqlen=args.seqlen)

    start_time = time.time()
    perplexity = evaluate_perplexity(model, wikitext2_data, batch_size=args.batch_size)
    eval_time = time.time() - start_time

    logger.info(f"WikiText-2 Perplexity: {perplexity:.4f}")
    logger.info(f"Evaluation time: {eval_time:.2f}s")

    results = {
        "model": args.model,
        "evaluation_type": "perplexity",
        "dataset": "wikitext-2",
        "wikitext2_perplexity": round(perplexity, 4),
        "nsamples": wikitext2_data.shape[0],
        "seqlen": args.seqlen,
        "batch_size": args.batch_size,
        "sparsity": sparsity,
        "model_size_mb": round(model_size_mb, 2),
        "total_parameters": total_params,
        "non_zero_parameters": non_zero_params,
        "evaluation_time_seconds": round(eval_time, 2),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    output_file = Path(args.output_file)
    save_json(results, output_file)

    logger.info("=" * 70)
    logger.info("Perplexity Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"WikiText-2 Perplexity: {perplexity:.4f}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    logger.info(f"Sparsity: {sparsity*100:.2f}%")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
