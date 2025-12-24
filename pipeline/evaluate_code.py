"""
Code benchmark evaluation script.

Evaluates code generation models on HumanEval and MBPP benchmarks
using bigcode-evaluation-harness.
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
    CODE_EVAL_CONFIG,
    setup_logging,
    ensure_paths_exist,
)
from utils.io_utils import save_json
from lib.eval.code_eval import evaluate_humaneval, evaluate_mbpp, run_bigcode_eval

logger = logging.getLogger(__name__)

# Available code benchmarks
AVAILABLE_TASKS = [
    "humaneval",  # - HumanEval benchmark
    "humanevalplus",  # - HumanEval+ with more rigorous tests
    "mbpp",  # - MBPP benchmark
    "mbppplus",  # - MBPP+ with more rigorous tests
    "multiple-py",  # - MultiPL-E Python
    "multiple-java",  # - MultiPL-E Java
    "multiple-cpp",  # - MultiPL-E C++
    "multiple-js",  # - MultiPL-E JavaScript
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on code generation benchmarks (HumanEval, MBPP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_CONFIG['name'],
        help='Model path or HuggingFace model name',
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=CODE_EVAL_CONFIG['tasks'],
        help=f'Code benchmarks to evaluate on. Available: {AVAILABLE_TASKS}',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=CODE_EVAL_CONFIG['n_samples'],
        help='Number of samples per problem (for pass@k evaluation)',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=CODE_EVAL_CONFIG['batch_size'],
        help='Batch size for generation',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=CODE_EVAL_CONFIG['temperature'],
        help='Sampling temperature (lower = more deterministic)',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=CODE_EVAL_CONFIG['top_p'],
        help='Top-p nucleus sampling parameter',
    )
    parser.add_argument(
        '--max_length_generation',
        type=int,
        default=CODE_EVAL_CONFIG['max_length_generation'],
        help='Maximum length of generation (prompt + completion)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of problems (for debugging/quick tests)',
    )
    parser.add_argument(
        '--precision',
        type=str,
        default=CODE_EVAL_CONFIG['precision'],
        choices=['fp32', 'fp16', 'bf16'],
        help='Model precision',
    )
    parser.add_argument(
        '--load_in_8bit',
        action='store_true',
        help='Load model in 8-bit quantization',
    )
    parser.add_argument(
        '--load_in_4bit',
        action='store_true',
        help='Load model in 4-bit quantization',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=PATHS_CONFIG['stats_dir'],
        help='Directory to save results.',
    )
    parser.add_argument(
        '--save_generations',
        action='store_true',
        default=True,
        help='Save generated code solutions',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        default=True,
        help='Trust remote code for custom models',
    )
    parser.add_argument(
        '--use_auth_token',
        action='store_true',
        help='Use HuggingFace auth token (from huggingface-cli login)',
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace auth token for private/gated models (alternative to --use_auth_token)',
    )
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging except errors")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    ensure_paths_exist()

    torch.manual_seed(args.seed)

    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Samples per problem: {args.n_samples}")
    logger.info(f"Temperature: {args.temperature}")

    all_results = {
        "model": args.model,
        "evaluation_type": "code_generation",
        "tasks": args.tasks,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "precision": args.precision,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    total_start_time = time.time()

    for task in args.tasks:
        logger.info("=" * 70)
        logger.info(f"Evaluating on {task}...")
        logger.info("=" * 70)

        output_path = f"{args.output_dir}/{task}_results_{int(time.time())}.json"

        start_time = time.time()

        # Use specialized functions for common tasks
        if task in ["humaneval", "humanevalplus"]:
            results = evaluate_humaneval(
                model_path=args.model,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_length_generation=args.max_length_generation,
                limit=args.limit,
                precision=args.precision,
                output_path=output_path,
                use_plus=(task == "humanevalplus"),
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                hf_token=args.hf_token,
                save_generations=args.save_generations,
                top_p=args.top_p,
            )
        elif task in ["mbpp", "mbppplus"]:
            results = evaluate_mbpp(
                model_path=args.model,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_length_generation=args.max_length_generation,
                limit=args.limit,
                precision=args.precision,
                output_path=output_path,
                use_plus=(task == "mbppplus"),
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                hf_token=args.hf_token,
                save_generations=args.save_generations,
                top_p=args.top_p,
            )
        else:
            # Use generic bigcode eval for other tasks
            results = run_bigcode_eval(
                model_path=args.model,
                tasks=task,
                output_path=output_path,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_length_generation=args.max_length_generation,
                limit=args.limit,
                precision=args.precision,
                load_in_8bit=args.load_in_8bit,
                load_in_4bit=args.load_in_4bit,
                trust_remote_code=args.trust_remote_code,
                use_auth_token=args.use_auth_token,
                hf_token=args.hf_token,
                save_generations=args.save_generations,
                top_p=args.top_p,
            )

        eval_time = time.time() - start_time

        # Extract and log pass@k metrics
        if task in results:
            task_results = results[task]
            pass_at_k = {k: v for k, v in task_results.items() if k.startswith("pass@")}
            for metric, score in pass_at_k.items():
                logger.info(f"{task} {metric}: {score:.4f} ({score*100:.2f}%)")
            all_results[task] = pass_at_k
        else:
            all_results[task] = results

        all_results[f"{task}_eval_time_seconds"] = round(eval_time, 2)
        logger.info(f"Evaluation time: {eval_time:.2f}s")

    total_time = time.time() - total_start_time
    all_results["total_eval_time_seconds"] = round(total_time, 2)

    combined_output = f"{args.output_dir}/code_eval_results_{int(time.time())}.json"
    save_json(all_results, Path(combined_output))
    logger.info(f"Combined results saved to: {combined_output}")

    logger.info("=" * 70)
    logger.info("Code Evaluation Summary")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    for task in args.tasks:
        if task in all_results and isinstance(all_results[task], dict):
            pass1 = all_results[task].get("pass@1", "N/A")
            logger.info(f"{task} pass@1: {pass1}")
    logger.info(f"Total evaluation time: {total_time:.2f}s")
    logger.info("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
