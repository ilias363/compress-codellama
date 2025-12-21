import logging
from typing import Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_wikitext2(tokenizer, seqlen: int = 2048, split: str = "test") -> torch.Tensor:
    """
    Load and tokenize WikiText-2 dataset.

    Args:
        tokenizer: Tokenizer to use
        seqlen: Sequence length for each sample
        split: Dataset split ("train", "validation", "test")

    Returns:
        Tensor of shape (nsamples, seqlen) containing tokenized input_ids
    """
    from datasets import load_dataset

    logger.info(f"Loading WikiText-2 {split} split...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    input_ids = encodings.input_ids[0]

    # Create chunks of seqlen
    nsamples = input_ids.numel() // seqlen
    input_ids = input_ids[: nsamples * seqlen].view(nsamples, seqlen)

    logger.info(f"WikiText-2: {nsamples} samples of length {seqlen}")
    return input_ids


@torch.no_grad()
def evaluate_perplexity(
    model,
    input_ids: torch.Tensor,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> float:
    """
    Evaluate perplexity of model on given input_ids.

    Compatible with Wanda/SparseGPT evaluation methodology.

    Args:
        model: The model to evaluate (must have model.seqlen attribute)
        input_ids: Tensor of shape (nsamples, seqlen) or raw input_ids from tokenizer
        batch_size: Batch size for evaluation
        device: Device to use (auto-detected if None)

    Returns:
        Perplexity score
    """
    model.eval()

    # Get device from model if not provided
    if device is None:
        device = next(model.parameters()).device
        if hasattr(model, 'hf_device_map'):
            if "lm_head" in model.hf_device_map:
                device = model.hf_device_map["lm_head"]

    # Pre-chunked format (nsamples, seqlen)
    nsamples = input_ids.shape[0]
    seqlen = input_ids.shape[1]

    nlls = []
    logger.info(f"nsamples {nsamples}")

    for i in tqdm(range(0, nsamples, batch_size), desc="Evaluating"):
        j = min(i + batch_size, nsamples)

        # Get batch
        inputs = input_ids[i:j].to(device)

        # Forward pass
        lm_logits = model(inputs).logits

        # Shift for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Calculate loss (mean reduction)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j - i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    torch.cuda.empty_cache()

    return ppl.item()
