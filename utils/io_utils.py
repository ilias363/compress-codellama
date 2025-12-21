import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


def save_json(data: Any, output_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info(f"Saved JSON to {output_path}")


def load_json(input_path: Union[str, Path]) -> Any:
    """Load data from a JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {input_path}")
    return data


def save_jsonl(data: List[Dict], output_path: Union[str, Path]) -> None:
    """Save data to a JSONL (JSON Lines) file."""
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(data)} records to {output_path}")


def load_jsonl(input_path: Union[str, Path]) -> List[Dict]:
    """Load data from a JSONL (JSON Lines) file."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} records from {input_path}")
    return data


def save_dataset(samples: List[Dict], output_path: Union[str, Path], format: str = 'json') -> None:
    """Save dataset to file with optional format selection."""
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'jsonl':
        save_jsonl(samples, output_path)
    else:
        save_json(samples, output_path)

    logger.info(f"Saved {len(samples)} samples to {output_path}")


def load_dataset_file(input_path: Union[str, Path]) -> List[Dict]:
    """Load dataset from file, auto-detecting format."""
    input_path = Path(input_path)

    if input_path.suffix == '.jsonl':
        return load_jsonl(input_path)
    else:
        return load_json(input_path)
