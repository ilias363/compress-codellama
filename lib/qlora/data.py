"""
QLoRA Data Module
Adapted from: https://github.com/artidoro/qlora
"""

import copy
import logging
from dataclasses import dataclass
from typing import Dict, Sequence, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from datasets import Dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# Code instruction prompt template for CodeLlama
CODE_INSTRUCT_PROMPT = {
    "prompt_with_input": ("### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"),
    "prompt_no_input": ("### Instruction:\n{instruction}\n\n### Response:\n"),
}


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling with instruction tuning.

    Handles tokenization and padding of input-output pairs for training.
    Masks the input portion of labels so loss is only computed on outputs.
    """

    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool = False
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements with BOS/EOS tokens
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources['input_ids'], tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    # Mask source tokens in labels
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict


def format_code_instruction(example: Dict) -> Dict:
    """
    Format a code instruction example into input/output format.

    Supports both instruction-only and instruction+input formats.
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    if input_text:
        formatted_input = CODE_INSTRUCT_PROMPT["prompt_with_input"].format(
            instruction=instruction, input=input_text
        )
    else:
        formatted_input = CODE_INSTRUCT_PROMPT["prompt_no_input"].format(instruction=instruction)

    return {'input': formatted_input, 'output': output}


def load_local_dataset(dataset_path: str) -> Dataset:
    """Load a local json or jsonl dataset."""
    if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    return full_dataset


def prepare_dataset(
    dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    source_max_len: int = 1024,
    target_max_len: int = 512,
    max_train_samples: Optional[int] = None,
    eval_dataset_size: int = 0,
    train_on_source: bool = False,
    group_by_length: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Prepare dataset for QLoRA training.

    Args:
        dataset: The raw dataset with 'instruction', 'input', 'output' columns
        tokenizer: The tokenizer to use
        source_max_len: Maximum length for source (instruction + input)
        target_max_len: Maximum length for target (output)
        max_train_samples: Maximum number of training samples (for debugging)
        eval_dataset_size: Size of validation set (split from train if > 0)
        train_on_source: Whether to train on source text as well as target
        group_by_length: Group sequences by length for efficient batching
        seed: Random seed for splitting

    Returns:
        Dict with train_dataset, eval_dataset, and data_collator
    """
    # Format the dataset
    logger.info("Formatting dataset for instruction tuning...")

    # Check if dataset needs formatting (has 'instruction' column)
    if 'instruction' in dataset.column_names:
        dataset = dataset.map(format_code_instruction)

    required_cols = ['input', 'output']
    for col in required_cols:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset must have '{col}' column. Found: {dataset.column_names}")

    cols_to_remove = [col for col in dataset.column_names if col not in ['input', 'output']]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    train_dataset = dataset
    eval_dataset = None

    if eval_dataset_size > 0:
        logger.info(f'Splitting dataset: {eval_dataset_size} samples for validation')
        split = dataset.train_test_split(test_size=eval_dataset_size, shuffle=True, seed=seed)
        train_dataset = split['train']
        eval_dataset = split['test']

    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        logger.info(f'Limiting training samples to {max_train_samples}')
        train_dataset = train_dataset.select(range(max_train_samples))

    if group_by_length:
        train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=source_max_len,
        target_max_len=target_max_len,
        train_on_source=train_on_source,
        predict_with_generate=False,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation samples: {len(eval_dataset)}")

    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator,
    }
