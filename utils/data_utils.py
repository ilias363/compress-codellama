import re
import random
import logging
from typing import List, Dict
from collections import Counter, defaultdict
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Language detection patterns
LANGUAGE_PATTERNS = {
    'python': [r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import', r'print\s*\('],
    'javascript': [r'\bfunction\s+\w+', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=', r'=>', r'console\.log'],
    'java': [r'\bpublic\s+class', r'\bprivate\s+\w+', r'System\.out\.println', r'\bvoid\s+\w+\s*\('],
    'cpp': [r'#include\s*<', r'\bstd::', r'\bint\s+main\s*\(', r'cout\s*<<', r'\btemplate\s*<'],
    'typescript': [
        r':\s*(string|number|boolean|any)\b',
        r'\binterface\s+\w+',
        r'\btype\s+\w+\s*=',
        r'export\s+(interface|type|class)',
    ],
}


class DatasetStats:
    """Track dataset statistics during processing."""

    def __init__(self):
        self.total_loaded = 0
        self.total_filtered = 0
        self.by_source = Counter()
        self.by_language = Counter()
        self.token_lengths = []

    def summary(self) -> str:
        """Generate a human-readable summary of statistics."""
        avg_tokens = sum(self.token_lengths) / len(self.token_lengths) if self.token_lengths else 0
        return f"""
Dataset Statistics:
==================
Total loaded:        {self.total_loaded}
After filtering:     {self.total_filtered}
Avg tokens:          {avg_tokens:.1f}

By Source:
{chr(10).join(f'  {k}: {v}' for k, v in self.by_source.most_common())}

By Language (detected):
{chr(10).join(f'  {k}: {v}' for k, v in self.by_language.most_common(10))}
"""

    def to_dict(self) -> Dict:
        """Convert statistics to dictionary for serialization."""
        return {
            'total_loaded': self.total_loaded,
            'total_filtered': self.total_filtered,
            'avg_token_length': (
                sum(self.token_lengths) / len(self.token_lengths) if self.token_lengths else 0
            ),
            'by_source': dict(self.by_source),
            'by_language': dict(self.by_language),
        }


def detect_language(code: str) -> str:
    """Detect programming language from code content using pattern matching.

    Returns:
        Detected language name or 'unknown'
    """
    scores = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, code))
        if score > 0:
            scores[lang] = score
    return max(scores, key=scores.get) if scores else 'unknown'


def load_magicoder(num_samples: int, languages: List[str], stats: DatasetStats) -> List[Dict]:
    """Load Magicoder-OSS-Instruct dataset."""
    logger.info(f"Loading Magicoder-OSS-Instruct ({num_samples} samples)...")
    try:
        dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")

        dataset_list = list(dataset)
        random.shuffle(dataset_list)

        samples = []
        for item in tqdm(dataset_list, desc="Processing Magicoder"):
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            lang = item.get("lang", "").lower()

            if languages and lang and lang not in languages:
                continue

            if lang:
                stats.by_language[lang] += 1

            samples.append({"instruction": problem, "input": "", "output": solution, "source": "magicoder"})
            stats.by_source['magicoder'] += 1

            if len(samples) >= num_samples:
                break

        logger.info(f"  Loaded {len(samples)} samples from Magicoder")
        return samples
    except Exception as e:
        logger.error(f"  Error loading Magicoder: {e}")
        return []


def load_wizardcoder(num_samples: int, languages: List[str], stats: DatasetStats) -> List[Dict]:
    """Load WizardCoder/WizardLM Evol-Instruct dataset (code-heavy subset)."""
    logger.info(f"Loading WizardCoder-Evol-Instruct ({num_samples} samples)...")
    try:
        dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train")

        dataset_list = list(dataset)
        random.shuffle(dataset_list)

        samples = []
        code_keywords = [
            "code",
            "function",
            "program",
            "algorithm",
            "implement",
            "python",
            "write a",
            "script",
            "method",
            "class",
            "module",
            "library",
            "api",
            "debug",
            "fix",
            "error",
            "bug",
            "optimize",
            "refactor",
        ]

        if languages:
            code_keywords.extend(languages)

        for item in tqdm(dataset_list, desc="Processing WizardCoder"):
            instruction = (
                item.get("conversations", [{}])[0].get("value", "") if item.get("conversations") else ""
            )
            output = (
                item.get("conversations", [{}] * 2)[1].get("value", "")
                if len(item.get("conversations", [])) > 1
                else ""
            )

            instruction_lower = instruction.lower()
            if not any(kw in instruction_lower for kw in code_keywords):
                continue

            # Filter by language if specified
            if languages:
                detected_lang = detect_language(output)
                if detected_lang not in languages and detected_lang != 'unknown':
                    continue
                stats.by_language[detected_lang] += 1
            else:
                detected_lang = detect_language(output)
                stats.by_language[detected_lang] += 1

            samples.append(
                {"instruction": instruction, "input": "", "output": output, "source": "wizardcoder"}
            )
            stats.by_source['wizardcoder'] += 1

            if len(samples) >= num_samples:
                break

        logger.info(f"  Loaded {len(samples)} samples from WizardCoder")
        return samples
    except Exception as e:
        logger.error(f"  Error loading WizardCoder: {e}")
        return []


def load_code_alpaca(num_samples: int, languages: List[str], stats: DatasetStats) -> List[Dict]:
    """Load Code Alpaca dataset."""
    logger.info(f"Loading Code Alpaca ({num_samples} samples)...")
    try:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

        dataset_list = list(dataset)
        random.shuffle(dataset_list)

        samples = []
        for item in tqdm(dataset_list, desc="Processing CodeAlpaca"):
            output = item.get("output", "")

            if languages:
                detected_lang = detect_language(output)
                if detected_lang not in languages and detected_lang != 'unknown':
                    continue
                stats.by_language[detected_lang] += 1
            else:
                detected_lang = detect_language(output)
                stats.by_language[detected_lang] += 1

            samples.append(
                {
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": output,
                    "source": "code_alpaca",
                }
            )
            stats.by_source['code_alpaca'] += 1

            if len(samples) >= num_samples:
                break

        logger.info(f"  Loaded {len(samples)} samples from Code Alpaca")
        return samples
    except Exception as e:
        logger.error(f"  Error loading Code Alpaca: {e}")
        return []


SOURCE_LOADERS = {
    "magicoder": load_magicoder,
    "wizardcoder": load_wizardcoder,
    "code_alpaca": load_code_alpaca,
}


def filter_by_length(
    samples: List[Dict], tokenizer, max_length: int, min_length: int, stats: DatasetStats
) -> List[Dict]:
    """Filter samples by token length."""
    logger.info(f"Filtering by length ({min_length}-{max_length} tokens)...")
    filtered = []

    for sample in tqdm(samples, desc="Filtering by length"):
        text = f"{sample['instruction']}\n{sample.get('input', '')}\n{sample['output']}"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)

        if min_length <= token_count <= max_length:
            sample['token_count'] = token_count
            filtered.append(sample)
            stats.token_lengths.append(token_count)

    logger.info(f"  Kept {len(filtered)} / {len(samples)} samples")
    return filtered


def create_calibration_subset(
    samples: List[Dict], num_samples: int, tokenizer, max_length: int = 512, stratified: bool = True
) -> List[Dict]:
    """
    Create calibration subset with shorter sequences.

    Args:
        samples: Source samples to select from
        num_samples: Number of calibration samples to create
        tokenizer: HuggingFace tokenizer
        max_length: Maximum token length for calibration samples
        stratified: If True, ensures diversity across sources

    Returns:
        List of calibration samples
    """
    logger.info(f"Creating calibration subset ({num_samples} samples, stratified={stratified})...")

    # Prefer shorter sequences for calibration
    scored_samples = []
    for sample in samples:
        text = f"{sample['instruction']}\n{sample.get('input', '')}\n{sample['output']}"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_length:
            scored_samples.append(
                {
                    'sample': sample,
                    'length': len(tokens),
                    'source': sample.get('source', 'unknown'),
                }
            )

    if not scored_samples:
        logger.warning("No samples fit within calibration max_length!")
        return []

    if stratified and len(scored_samples) >= num_samples:
        by_source = defaultdict(list)
        for item in scored_samples:
            by_source[item['source']].append(item)

        n_sources = len(by_source)
        base_per_source = num_samples // n_sources

        calibration = []
        for source, items in by_source.items():
            items.sort(key=lambda x: x['length'])
            # Take evenly spaced samples
            n_take = min(base_per_source, len(items))
            step = max(1, len(items) // n_take)
            for i in range(0, len(items), step):
                if len(calibration) < num_samples:
                    calibration.append(items[i]['sample'])

        # Fill remaining slots
        random.shuffle(scored_samples)
        for item in scored_samples:
            if len(calibration) >= num_samples:
                break
            if item['sample'] not in calibration:
                calibration.append(item['sample'])
    else:
        # Non-stratified: sort by length and take diverse subset
        scored_samples.sort(key=lambda x: x['length'])
        step = len(scored_samples) // num_samples if len(scored_samples) > num_samples else 1
        calibration = [scored_samples[i]['sample'] for i in range(0, len(scored_samples), step)][:num_samples]

    logger.info(f"  Created calibration set with {len(calibration)} samples")
    return calibration


def create_awq_calibration(samples: List[Dict], num_samples: int = 128) -> List[Dict]:
    """Create smaller calibration set specifically for AWQ quantization."""
    logger.info(f"Creating AWQ calibration subset ({num_samples} samples)...")
    shuffled = samples.copy()
    random.shuffle(shuffled)
    return shuffled[:num_samples]
