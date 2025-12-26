"""
Code evaluation using bigcode-evaluation-harness.

This module provides wrapper functions to run HumanEval, MBPP, and other
code benchmarks using the bigcode-evaluation-harness library.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

# Path to bigcode-evaluation-harness
BIGCODE_HARNESS_PATH = Path(__file__).resolve().parent.parent.parent / "bigcode-evaluation-harness"


def run_bigcode_eval(
    model_path: str,
    tasks: Union[str, List[str]],
    output_path: Optional[str] = None,
    n_samples: int = 1,
    batch_size: int = 1,
    temperature: float = 0.2,
    max_length_generation: int = 512,
    limit: Optional[int] = None,
    precision: str = "bf16",
    allow_code_execution: bool = True,
    trust_remote_code: bool = True,
    save_generations: bool = True,
    save_generations_path: str = "generations.json",
    save_references: bool = True,
    save_references_path: str = "references.json",
    use_auth_token: bool = False,
    hf_token: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    do_sample: bool = True,
    top_p: float = 0.95,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run bigcode-evaluation-harness for code benchmarks.

    Args:
        model_path: Path to model or HuggingFace model name
        tasks: Task(s) to evaluate on (e.g., "humaneval", "mbpp", "humaneval,mbpp")
        output_path: Path to save evaluation results
        n_samples: Number of samples per problem (for pass@k)
        batch_size: Batch size for generation
        temperature: Sampling temperature
        max_length_generation: Maximum generation length
        limit: Limit number of problems (for debugging)
        precision: Model precision (fp32, fp16, bf16)
        allow_code_execution: Allow execution of generated code
        trust_remote_code: Trust remote code for custom models
        save_generations: Save generated solutions
        save_generations_path: Path for saving the code generations
        save_references: Whether to save reference solutions/tests
        save_references_path: Path for saving the references solutions/tests
        use_auth_token: Use HuggingFace auth token (from login)
        hf_token: Explicit HuggingFace token for private/gated models
        load_in_8bit: Load model in 8-bit quantization
        load_in_4bit: Load model in 4-bit quantization
        do_sample: Use sampling for generation
        top_p: Top-p nucleus sampling
        extra_args: Additional command-line arguments

    Returns:
        Dictionary with evaluation results
    """
    if isinstance(tasks, list):
        tasks = ",".join(tasks)

    if output_path is None:
        output_path = f"evaluation_results_{tasks.replace(',', '_')}.json"

    # Convert output_path to absolute path to avoid issues with cwd
    output_path = str(Path(output_path).resolve())

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    save_generations_path = str(output_dir / save_generations_path)
    save_references_path = str(output_dir / save_references_path)

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        str(BIGCODE_HARNESS_PATH / "main.py"),
        "--model",
        model_path,
        "--tasks",
        tasks,
        "--n_samples",
        str(n_samples),
        "--batch_size",
        str(batch_size),
        "--temperature",
        str(temperature),
        "--max_length_generation",
        str(max_length_generation),
        "--precision",
        precision,
        "--metric_output_path",
        output_path,
        "--top_p",
        str(top_p),
    ]

    if do_sample:
        cmd.extend(["--do_sample", "True"])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if allow_code_execution:
        cmd.append("--allow_code_execution")

    if trust_remote_code:
        cmd.append("--trust_remote_code")

    if save_generations:
        cmd.append("--save_generations")
        cmd.extend(["--save_generations_path", save_generations_path])

    if save_references:
        cmd.append("--save_references")
        cmd.extend(["--save_references_path", save_references_path])

    if use_auth_token or hf_token:
        cmd.append("--use_auth_token")

    if load_in_8bit:
        cmd.append("--load_in_8bit")

    if load_in_4bit:
        cmd.append("--load_in_4bit")

    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running bigcode-evaluation-harness: {' '.join(cmd)}")

    # Set environment to include bigcode-evaluation-harness in path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BIGCODE_HARNESS_PATH) + os.pathsep + env.get("PYTHONPATH", "")

    # Set HF token in environment if provided explicitly
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Run the evaluation
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BIGCODE_HARNESS_PATH),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed: {e.stderr}")
        raise RuntimeError(f"bigcode-evaluation-harness failed: {e.stderr}")

    # Load and return results
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return json.load(f)
    else:
        logger.warning(f"Results file not found: {output_path}")
        return {}


def evaluate_humaneval(
    model_path: str,
    n_samples: int = 1,
    batch_size: int = 1,
    temperature: float = 0.2,
    max_length_generation: int = 512,
    limit: Optional[int] = None,
    precision: str = "bf16",
    output_path: Optional[str] = None,
    allow_code_execution: bool = True,
    use_plus: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate model on HumanEval benchmark.

    Args:
        model_path: Path to model or HuggingFace model name
        n_samples: Number of samples per problem (for pass@k)
        batch_size: Batch size for generation
        temperature: Sampling temperature
        max_length_generation: Maximum generation length
        limit: Limit number of problems (for debugging)
        precision: Model precision (fp32, fp16, bf16)
        output_path: Path to save results
        allow_code_execution: Allow code execution
        use_plus: Use HumanEval+ (harder version with more tests)
        **kwargs: Additional arguments passed to run_bigcode_eval

    Returns:
        Dictionary with pass@k scores and detailed results
    """
    task = "humanevalplus" if use_plus else "humaneval"

    if output_path is None:
        output_path = f"humaneval_results.json"

    logger.info(f"Evaluating on {task}...")

    results = run_bigcode_eval(
        model_path=model_path,
        tasks=task,
        output_path=output_path,
        n_samples=n_samples,
        batch_size=batch_size,
        temperature=temperature,
        max_length_generation=max_length_generation,
        limit=limit,
        precision=precision,
        allow_code_execution=allow_code_execution,
        **kwargs,
    )

    return results


def evaluate_mbpp(
    model_path: str,
    n_samples: int = 1,
    batch_size: int = 1,
    temperature: float = 0.2,
    max_length_generation: int = 512,
    limit: Optional[int] = None,
    precision: str = "bf16",
    output_path: Optional[str] = None,
    allow_code_execution: bool = True,
    use_plus: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate model on MBPP benchmark.

    Args:
        model_path: Path to model or HuggingFace model name
        n_samples: Number of samples per problem (for pass@k)
        batch_size: Batch size for generation
        temperature: Sampling temperature
        max_length_generation: Maximum generation length
        limit: Limit number of problems (for debugging)
        precision: Model precision (fp32, fp16, bf16)
        output_path: Path to save results
        allow_code_execution: Allow code execution
        use_plus: Use MBPP+ (harder version with more tests)
        **kwargs: Additional arguments passed to run_bigcode_eval

    Returns:
        Dictionary with pass@k scores and detailed results
    """
    task = "mbppplus" if use_plus else "mbpp"

    if output_path is None:
        output_path = f"mbpp_results.json"

    logger.info(f"Evaluating on {task}...")

    results = run_bigcode_eval(
        model_path=model_path,
        tasks=task,
        output_path=output_path,
        n_samples=n_samples,
        batch_size=batch_size,
        temperature=temperature,
        max_length_generation=max_length_generation,
        limit=limit,
        precision=precision,
        allow_code_execution=allow_code_execution,
        **kwargs,
    )

    return results


def evaluate_multiple_benchmarks(
    model_path: str,
    tasks: List[str] = ["humaneval", "mbpp"],
    n_samples: int = 1,
    batch_size: int = 1,
    temperature: float = 0.2,
    max_length_generation: int = 512,
    limit: Optional[int] = None,
    precision: str = "bf16",
    output_path: Optional[str] = None,
    allow_code_execution: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate model on multiple code benchmarks at once.

    Args:
        model_path: Path to model or HuggingFace model name
        tasks: List of tasks to evaluate on
        n_samples: Number of samples per problem
        batch_size: Batch size for generation
        temperature: Sampling temperature
        max_length_generation: Maximum generation length
        limit: Limit number of problems per task
        precision: Model precision
        output_path: Path to save results
        allow_code_execution: Allow code execution
        **kwargs: Additional arguments

    Returns:
        Dictionary with results for each task
    """
    if output_path is None:
        output_path = "multi_benchmark_results.json"

    logger.info(f"Evaluating on tasks: {tasks}")

    results = run_bigcode_eval(
        model_path=model_path,
        tasks=tasks,
        output_path=output_path,
        n_samples=n_samples,
        batch_size=batch_size,
        temperature=temperature,
        max_length_generation=max_length_generation,
        limit=limit,
        precision=precision,
        allow_code_execution=allow_code_execution,
        **kwargs,
    )

    return results
