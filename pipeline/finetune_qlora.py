"""
QLoRA Fine-tuning Pipeline for CodeLlama
Adapted from: https://github.com/artidoro/qlora
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_utils import (
    SEED,
    PATHS_CONFIG,
    DATASET_CONFIG,
    FINETUNING_CONFIG,
    setup_logging,
    ensure_paths_exist,
)
from utils.model_utils import get_trainable_parameters, get_dtype_distribution
from utils.io_utils import save_json
from lib.qlora.trainer import get_qlora_model, SavePeftModelCallback, get_last_checkpoint
from lib.qlora.data import load_local_dataset, prepare_dataset


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for pruned CodeLlama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/pruned",
        help='Base model path (pruned model) or HuggingFace model name',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=f"{PATHS_CONFIG['cache_dir']}/llm_weights",
        help='Cache directory for model weights',
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace auth token for accessing private/gated models',
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default=f"{PATHS_CONFIG['datasets_dir']}/{DATASET_CONFIG['train_file']}",
        help='Path to training dataset (JSON/JSONL)',
    )
    parser.add_argument(
        '--source_max_len',
        type=int,
        default=FINETUNING_CONFIG['source_max_len'],
        help='Maximum source (instruction+input) sequence length',
    )
    parser.add_argument(
        '--target_max_len',
        type=int,
        default=FINETUNING_CONFIG['target_max_len'],
        help='Maximum target (output) sequence length',
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=FINETUNING_CONFIG['max_train_samples'],
        help='Maximum number of training samples (for debugging)',
    )
    parser.add_argument(
        '--eval_dataset_size',
        type=int,
        default=FINETUNING_CONFIG['eval_dataset_size'],
        help='Number of samples for validation set',
    )

    # Quantization arguments
    parser.add_argument(
        '--bits',
        type=int,
        default=FINETUNING_CONFIG['bits'],
        choices=[4, 8],
        help='Quantization bits for base model (4 or 8)',
    )
    parser.add_argument(
        '--double_quant',
        action='store_true',
        default=FINETUNING_CONFIG['double_quant'],
        help='Use double quantization for additional memory savings',
    )
    parser.add_argument(
        '--quant_type',
        type=str,
        default=FINETUNING_CONFIG['quant_type'],
        choices=['nf4', 'fp4'],
        help='Quantization type (nf4 is recommended)',
    )

    # LoRA arguments
    parser.add_argument(
        '--lora_r',
        type=int,
        default=FINETUNING_CONFIG['lora_r'],
        help='LoRA rank dimension (higher = more capacity but more parameters)',
    )
    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=FINETUNING_CONFIG['lora_alpha'],
        help='LoRA alpha scaling factor',
    )
    parser.add_argument(
        '--lora_dropout',
        type=float,
        default=FINETUNING_CONFIG['lora_dropout'],
        help='Dropout for LoRA layers (regularization)',
    )

    # Training arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default=f"{PATHS_CONFIG['models_dir']}/finetuned",
        help='Directory to save fine-tuned model',
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=FINETUNING_CONFIG['num_train_epochs'],
        help='Number of training epochs',
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=FINETUNING_CONFIG['max_steps'],
        help='Maximum training steps (-1 for epoch-based training)',
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=FINETUNING_CONFIG['per_device_train_batch_size'],
        help='Training batch size per GPU',
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=FINETUNING_CONFIG['per_device_eval_batch_size'],
        help='Evaluation batch size per GPU',
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=FINETUNING_CONFIG['gradient_accumulation_steps'],
        help='Gradient accumulation steps (effective batch = batch_size * accumulation)',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=FINETUNING_CONFIG['learning_rate'],
        help='Learning rate for AdamW optimizer',
    )
    parser.add_argument(
        '--lr_scheduler_type',
        type=str,
        default=FINETUNING_CONFIG['lr_scheduler_type'],
        help='Learning rate scheduler type',
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=FINETUNING_CONFIG['warmup_ratio'],
        help='Warmup ratio for learning rate scheduler',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=FINETUNING_CONFIG['weight_decay'],
        help='Weight decay (L2 regularization)',
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=FINETUNING_CONFIG['max_grad_norm'],
        help='Maximum gradient norm for clipping',
    )

    # Memory optimization
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        default=FINETUNING_CONFIG['gradient_checkpointing'],
        help='Use gradient checkpointing to reduce memory usage',
    )
    parser.add_argument(
        '--max_memory_mb',
        type=int,
        default=FINETUNING_CONFIG['max_memory_mb'],
        help='Maximum GPU memory to use per device (MB)',
    )
    parser.add_argument(
        '--group_by_length',
        action='store_true',
        default=FINETUNING_CONFIG['group_by_length'],
        help='Group sequences by length for efficient batching',
    )

    # Precision
    parser.add_argument(
        '--bf16',
        action='store_true',
        default=FINETUNING_CONFIG['bf16'],
        help='Use bfloat16 precision (recommended for A100/H100)',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=FINETUNING_CONFIG['fp16'],
        help='Use float16 precision',
    )

    # Logging and saving
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=FINETUNING_CONFIG['logging_steps'],
        help='Log training metrics every N steps',
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=FINETUNING_CONFIG['save_steps'],
        help='Save checkpoint every N steps',
    )
    parser.add_argument(
        '--save_total_limit',
        type=int,
        default=FINETUNING_CONFIG['save_total_limit'],
        help='Maximum number of checkpoints to keep',
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=FINETUNING_CONFIG['eval_steps'],
        help='Evaluate every N steps',
    )
    parser.add_argument(
        '--stats_dir',
        type=str,
        default=PATHS_CONFIG['stats_dir'],
        help='Directory to save training statistics',
    )

    # Other
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument(
        '--resume_from_checkpoint',
        action='store_true',
        default=False,
        help='Resume training from last checkpoint',
    )
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

    logger.info(f"Setting random seed: {args.seed}")
    set_seed(args.seed)

    if not os.path.exists(args.dataset):
        logger.error(f"Training dataset not found: {args.dataset}")
        sys.exit(1)

    checkpoint_dir = None
    if args.resume_from_checkpoint:
        checkpoint_dir, completed = get_last_checkpoint(args.output_dir)
        if completed:
            logger.info('Training was already completed!')
            return
        if checkpoint_dir:
            logger.info(f'Resuming from checkpoint: {checkpoint_dir}')

    logger.info("=" * 70)
    logger.info("QLoRA Fine-tuning Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Base model: {args.model}")
    logger.info(f"  Training dataset: {args.dataset}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Quantization: {args.bits}-bit with {args.quant_type}")
    logger.info(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(
        f"  Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} accumulation"
    )
    logger.info(
        f"  Epochs: {args.num_train_epochs}" if args.max_steps < 0 else f"  Max steps: {args.max_steps}"
    )
    logger.info("=" * 70)

    logger.info("Loading model with QLoRA configuration...")

    model, tokenizer = get_qlora_model(
        model_name_or_path=args.model,
        cache_dir=args.cache_dir,
        bits=args.bits,
        double_quant=args.double_quant,
        quant_type=args.quant_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=args.gradient_checkpointing,
        max_memory_mb=args.max_memory_mb,
        checkpoint_dir=checkpoint_dir,
        bf16=args.bf16,
        fp16=args.fp16,
        use_auth_token=args.hf_token,
    )

    # Get parameter info
    param_info = get_trainable_parameters(model, bits=args.bits)
    dtype_info = get_dtype_distribution(model)
    logger.info(
        f"Trainable parameters: {param_info['trainable_params']:,} ({param_info['trainable_percent']:.2f}%)"
    )

    # Disable cache for training
    model.config.use_cache = False

    logger.info("Loading and preparing dataset...")
    dataset = load_local_dataset(args.dataset)

    data_module = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        max_train_samples=args.max_train_samples,
        eval_dataset_size=args.eval_dataset_size,
        group_by_length=args.group_by_length,
        seed=args.seed,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if data_module['eval_dataset'] else "no",
        eval_steps=args.eval_steps if data_module['eval_dataset'] else None,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        optim="paged_adamw_32bit",
        ddp_find_unused_parameters=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module['train_dataset'],
        eval_dataset=data_module['eval_dataset'],
        data_collator=data_module['data_collator'],
    )

    # Add PEFT save callback to save adapter weights during checkpoints and at training end
    trainer.add_callback(SavePeftModelCallback)

    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    train_start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
    training_time = time.time() - train_start_time

    logger.info(f"Training completed in {training_time:.2f}s")

    # Log and save metrics (following original qlora repo pattern)
    # Note: SavePeftModelCallback handles saving the adapter model on train_end
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate if eval dataset exists
    if data_module['eval_dataset']:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        metrics.update(eval_metrics)

    stats_path = Path(args.stats_dir) / f"finetuning_stats_{int(time.time())}.json"
    training_stats = {
        "method": "qlora",
        "model": args.model,
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "quantization": {
            "bits": args.bits,
            "double_quant": args.double_quant,
            "quant_type": args.quant_type,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "training": {
            "epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
        },
        "parameters": param_info,
        "dtype_distribution": dtype_info,
        "metrics": metrics,
        "training_time_seconds": round(training_time, 2),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    save_json(training_stats, stats_path)

    logger.info("=" * 70)
    logger.info("QLoRA fine-tuning completed successfully!")
    logger.info(f"Fine-tuned model saved to: {args.output_dir}")
    logger.info(f"Training statistics saved to: {stats_path}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
