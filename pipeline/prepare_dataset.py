import argparse
import random
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_utils import (
    PATHS_CONFIG,
    SEED,
    DATASET_CONFIG,
    MODEL_CONFIG,
    setup_logging,
    ensure_paths_exist,
)
from utils.data_utils import DatasetStats, SOURCE_LOADERS, create_calibration_subset, create_awq_calibration
from utils.model_utils import load_tokenizer
from utils.io_utils import save_dataset, save_json


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for CodeLLaMA compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DATASET_CONFIG['sources']),
        help="Comma-separated list of sources: magicoder,wizardcoder,code_alpaca",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=",".join(str(sample) for sample in DATASET_CONFIG['samples_per_source']),
        help="Comma-separated number of samples per source",
    )

    parser.add_argument(
        "--output_dir", type=str, default=PATHS_CONFIG['datasets_dir'], help="Output directory for datasets"
    )
    parser.add_argument(
        "--stats_dir", type=str, default=PATHS_CONFIG['stats_dir'], help="Output directory for statistics"
    )
    parser.add_argument(
        "--train_file", type=str, default=DATASET_CONFIG['train_file'], help="Training data filename"
    )
    parser.add_argument(
        "--calib_file", type=str, default=DATASET_CONFIG['calib_file'], help="Calibration data filename"
    )
    parser.add_argument(
        "--awq_calib_file",
        type=str,
        default=DATASET_CONFIG['awq_calib_file'],
        help="AWQ calibration data filename",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=DATASET_CONFIG['format'],
        choices=["json", "jsonl"],
        help="Output format",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=DATASET_CONFIG['max_length'],
        help="Maximum sequence length in tokens",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=DATASET_CONFIG['min_length'],
        help="Minimum sequence length in tokens",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=DATASET_CONFIG['calib_samples'],
        help="Number of calibration samples for Wanda",
    )
    parser.add_argument(
        "--awq_calib_samples",
        type=int,
        default=DATASET_CONFIG['awq_calib_samples'],
        help="Number of calibration samples for AWQ",
    )
    parser.add_argument(
        "--filter_languages",
        type=str,
        default=",".join(DATASET_CONFIG['languages']),
        help="Comma-separated list of languages to prioritize",
    )

    parser.add_argument(
        "--stratified_calib",
        action="store_true",
        default=DATASET_CONFIG['stratified_calib'],
        help="Use stratified sampling for calibration",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_CONFIG['name'],
        help="Model name for tokenizer",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")

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

    random.seed(args.seed)

    stats = DatasetStats()

    sources = args.sources.split(",")
    samples_per_source = [int(x) for x in args.samples.split(",")]

    while len(samples_per_source) < len(sources):
        samples_per_source.append(samples_per_source[-1] if samples_per_source else 5000)

    languages = [lang.lower() for lang in args.filter_languages.split(",")] if args.filter_languages else []
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Dataset Preparation for CodeLLaMA Compression")
    logger.info("=" * 60)
    logger.info(f"Sources: {sources}")
    logger.info(f"Samples per source: {samples_per_source}")
    logger.info(f"Languages: {languages}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    tokenizer = load_tokenizer(args.model_name)

    all_samples = []
    for source, num_samples in zip(sources, samples_per_source):
        loader = SOURCE_LOADERS.get(source.lower())
        if loader:
            samples = loader(
                num_samples,
                languages,
                stats,
                tokenizer=tokenizer,
                min_length=args.min_length,
                max_length=args.max_length,
            )
            all_samples.extend(samples)
            stats.total_loaded += len(samples)
        else:
            logger.warning(f"Unknown source: {source}")
            logger.info(f"Available sources: {list(SOURCE_LOADERS.keys())}")

    logger.info(f"\nTotal samples loaded: {len(all_samples)}")

    random.shuffle(all_samples)

    train_path = output_dir / args.train_file
    if args.format == 'jsonl':
        train_path = train_path.with_suffix('.jsonl')
    save_dataset(all_samples, train_path, args.format)

    logger.info(f"\nCreating Wanda calibration subset ({args.calib_samples} samples)...")
    calib_samples = create_calibration_subset(
        all_samples, args.calib_samples, tokenizer, max_length=512, stratified=args.stratified_calib
    )
    save_dataset(calib_samples, output_dir / args.calib_file, args.format)

    logger.info(f"Creating AWQ calibration subset ({args.awq_calib_samples} samples)...")
    awq_calib_samples = create_awq_calibration(calib_samples, args.awq_calib_samples)
    save_dataset(awq_calib_samples, output_dir / args.awq_calib_file, args.format)

    stats_path = Path(args.stats_dir) / f"dataset_stats{int(time.time())}.json"
    save_json(stats.to_dict(), stats_path)

    logger.info("=" * 60)
    logger.info("Dataset preparation complete!")
    logger.info("=" * 60)
    logger.info(f"  Training data:      {train_path} ({len(all_samples)} samples)")
    logger.info(f"  Wanda calibration:  {output_dir / args.calib_file} ({len(calib_samples)} samples)")
    logger.info(
        f"  AWQ calibration:    {output_dir / args.awq_calib_file} ({len(awq_calib_samples)} samples)"
    )


if __name__ == "__main__":
    main()
