import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.json"

DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    verbose: int = 0,
    quiet: bool = False,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_LOG_DATE_FORMAT,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG)
        quiet: If True, only show ERROR level logs
        log_format: Format string for log messages
        date_format: Format string for timestamps
        log_file: Optional file path to write logs to
    """
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a'))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    path = config_path or _CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {path}")
        return get_default_config()
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return hardcoded default configuration."""
    return {
        "seed": 363,
        "model": {"name": "codellama/CodeLlama-7b-Instruct-hf"},
        "paths": {
            "base_output_dir": "./outputs",
            "datasets_dir": "./datasets",
            "models_dir": "./outputs/models",
            "cache_dir": "./outputs/cache",
            "stats_dir": "./outputs/stats",
            "logs_dir": "./outputs/logs",
        },
        "dataset_prep": {
            "sources": ["magicoder", "wizardcoder", "code_alpaca"],
            "samples_per_source": [10000, 5000, 5000],
            "train_file": "train.json",
            "calib_file": "calib_512.json",
            "awq_calib_file": "calib_awq_128.json",
            "max_length": 1024,
            "min_length": 32,
            "calib_samples": 512,
            "awq_calib_samples": 128,
            "languages": ["python", "javascript", "java", "cpp", "typescript"],
            "stratified_calib": True,
            "format": "json",
        },
        "pruning": {
            "method": "wanda",
            "sparsity_ratio": 0.35,
            "sparsity_type": "unstructured",
            "nsamples": 512,
            "max_calib_seqlen": 512,
            "use_variant": False,
        },
        "evaluation": {
            "seqlen": 2048,
            "batch_size": 1,
        },
        "qlora": {},
        "quantization": {},
    }


# Load config at module import time
_FULL_CONFIG = load_config()

SEED: int = _FULL_CONFIG.get("seed", 363)
PATHS_CONFIG: Dict[str, Any] = _FULL_CONFIG.get("paths", get_default_config()["paths"])
DATASET_CONFIG: Dict[str, Any] = _FULL_CONFIG.get("dataset_prep", get_default_config()["dataset_prep"])
MODEL_CONFIG: Dict[str, Any] = _FULL_CONFIG.get("model", get_default_config()["model"])
PRUNING_CONFIG: Dict[str, Any] = _FULL_CONFIG.get("pruning", get_default_config()["pruning"])
EVALUATION_CONFIG: Dict[str, Any] = _FULL_CONFIG.get("evaluation", get_default_config()["evaluation"])
QLORA_CONFIG: Dict[str, Any] = _FULL_CONFIG.get("qlora", get_default_config()["qlora"])
QUANTIZATION_CONFIG: Dict[str, Any] = _FULL_CONFIG.get("quantization", get_default_config()["quantization"])


def get_config_section(section: str) -> Dict[str, Any]:
    """Get a specific section of the configuration."""
    sections = {
        "paths": PATHS_CONFIG,
        "dataset_prep": DATASET_CONFIG,
        "model": MODEL_CONFIG,
        "pruning": PRUNING_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "qlora": QLORA_CONFIG,
        "quantization": QUANTIZATION_CONFIG,
    }
    return sections.get(section, {})


def ensure_paths_exist() -> Dict[str, Path]:
    """Create all output directories if they don't exist.

    Returns:
        Dictionary mapping path keys to their Path objects
    """
    paths = {}
    for key in ["models_dir", "cache_dir", "stats_dir", "logs_dir"]:
        path = PATHS_CONFIG.get(key, f"./outputs/{key}")
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path

    # Also ensure datasets dir exists
    datasets_path = PATHS_CONFIG["datasets_dir"]
    datasets_path.mkdir(parents=True, exist_ok=True)
    paths["datasets_dir"] = datasets_path

    return paths


def reload_config(config_path: Optional[Path] = None) -> None:
    """
    Reload configuration from file. Updates global config variables.

    Args:
        config_path: Path to config file. Defaults to configs/default.json
    """
    global _FULL_CONFIG, SEED, PATHS_CONFIG, DATASET_CONFIG, MODEL_CONFIG, PRUNING_CONFIG, EVALUATION_CONFIG, QLORA_CONFIG, QUANTIZATION_CONFIG

    _FULL_CONFIG = load_config(config_path)
    SEED = _FULL_CONFIG["seed"]
    PATHS_CONFIG = _FULL_CONFIG["paths"]
    DATASET_CONFIG = _FULL_CONFIG["dataset_prep"]
    MODEL_CONFIG = _FULL_CONFIG["model"]
    PRUNING_CONFIG = _FULL_CONFIG["pruning"]
    EVALUATION_CONFIG = _FULL_CONFIG["evaluation"]
    QLORA_CONFIG = _FULL_CONFIG["qlora"]
    QUANTIZATION_CONFIG = _FULL_CONFIG["quantization"]
