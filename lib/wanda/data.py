"""
Data Loading Utilities for Wanda Pruning
Adapted from: https://github.com/locuslab/wanda
Modified to use custom JSON calibration datasets
"""

import json
import torch


def load_calibration_dataset(dataset_path, nsamples, seqlen, tokenizer):
    """
    Load calibration dataset from JSON file

    Args:
        dataset_path: Path to JSON dataset file
        nsamples: Number of samples to use
        seqlen: Sequence length
        tokenizer: Tokenizer to use

    Returns:
        trainloader: List of (input, target) tuples
    """
    print(f"Loading calibration dataset from {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trainloader = []
    sample_count = 0

    for item in data:
        if sample_count >= nsamples:
            break

        # Combine instruction, input, and output into a single text
        text_parts = []
        if item.get('instruction'):
            text_parts.append(f"Instruction: {item['instruction']}")
        if item.get('input'):
            text_parts.append(f"Input: {item['input']}")
        if item.get('output'):
            text_parts.append(f"Output: {item['output']}")

        text = "\n".join(text_parts)

        # Tokenize the text
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=seqlen)

        if enc.input_ids.shape[1] >= seqlen:
            inp = enc.input_ids[:, :seqlen]
        else:
            # Pad if needed
            padding_length = seqlen - enc.input_ids.shape[1]
            inp = torch.nn.functional.pad(enc.input_ids, (0, padding_length), value=tokenizer.pad_token_id)

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        sample_count += 1

    print(f"Loaded {len(trainloader)} samples from calibration dataset")
    return trainloader
