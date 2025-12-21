import argparse
import os
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import SEED, MODEL_CONFIG, setup_logging
from utils.model_utils import load_model, load_tokenizer, check_sparsity, get_model_size_mb, count_parameters
from utils.eval_utils import get_wikitext2, evaluate_perplexity
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model perplexity on WikiText-2 and C4",
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
        default='./outputs/cached_llm_weights',
        help='Cache directory for model weights',
    )
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length for evaluation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument(
        '--output_file', type=str, default=None, help='Path to save evaluation results (JSON)'
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging except errors")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)

    torch.manual_seed(args.seed)

    model = load_model(args.model, cache_dir=args.cache_dir, set_seqlen=True)
    tokenizer = load_tokenizer(args.model)

    sparsity = check_sparsity(model)
    model_size_mb = get_model_size_mb(model)
    total_params = count_parameters(model)
    non_zero_params = int(total_params * (1 - sparsity))

    logger.info(f"Model sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")

    results = {
        "model": args.model,
        "sparsity": sparsity,
        "model_size_mb": round(model_size_mb, 2),
        "total_parameters": total_params,
        "non_zero_parameters": non_zero_params,
        "seqlen": args.seqlen,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info("=" * 70)
    logger.info("Evaluating on WikiText-2...")
    logger.info("=" * 70)

    wikitext2_data = get_wikitext2(tokenizer, seqlen=args.seqlen)

    start_time = time.time()
    wikitext2_ppl = evaluate_perplexity(model, wikitext2_data, batch_size=args.batch_size)
    eval_time = time.time() - start_time

    logger.info(f"WikiText-2 Perplexity: {wikitext2_ppl:.4f}")
    logger.info(f"Evaluation time: {eval_time:.2f}s")

    results["wikitext2_perplexity"] = round(wikitext2_ppl, 4)
    results["wikitext2_nsamples"] = wikitext2_data.shape[0]
    results["evaluation_time_seconds"] = round(eval_time, 2)

    logger.info("=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Sparsity: {sparsity*100:.2f}%")
    logger.info(f"Non-zero parameters: {non_zero_params:,}")
    logger.info(f"WikiText-2 Perplexity: {wikitext2_ppl:.4f}")
    logger.info(f"Evaluation time: {eval_time:.2f}s")
    logger.info("=" * 70)

    if args.output_file:
        output_path = args.output_file
    else:
        # Default: save next to model
        if os.path.isdir(args.model):
            output_path = os.path.join(args.model, "eval_results.json")
        else:
            output_path = "./eval_results.json"

    save_json(results, output_path)

    return results


if __name__ == "__main__":
    main()
