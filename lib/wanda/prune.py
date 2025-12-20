"""
Wanda Pruning Implementation
Adapted from: https://github.com/locuslab/wanda
"""

import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT
from .data import load_calibration_dataset


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def check_sparsity(model):
    """
    Check and print the sparsity of each layer and overall model.

    Args:
        model: The model to check

    Returns:
        float: Overall sparsity ratio
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device, nsamples=128):
    """
    Prepare calibration inputs by passing data through the model.

    Args:
        model: The model
        dataloader: Calibration data loader
        device: Device to use
        nsamples: Number of calibration samples

    Returns:
        tuple: (inps, outs, attention_mask, position_ids)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Get device from model if available
    if hasattr(model, 'hf_device_map') and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    # Use a small sequence length for calibration to avoid OOM
    # 128 is commonly used in pruning papers and is sufficient for calibration
    calib_seqlen = min(model.seqlen, 128)

    dtype = next(iter(model.parameters())).dtype
    # Allocate on CPU first to avoid GPU OOM during allocation
    inps = torch.zeros((nsamples, calib_seqlen, model.config.hidden_size), dtype=dtype, device='cpu')
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # Copy to CPU to save GPU memory
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    """
    Helper function for Wanda variant pruning with adaptive threshold.

    Args:
        alpha: Threshold parameter
        sort_res: Sorted weight metrics
        W_metric: Weight metrics
        tmp_metric: Cumulative sum of sorted metrics
        sum_before: Sum of metrics before sorting

    Returns:
        tuple: (W_mask, cur_sparsity)
    """
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(
    model,
    tokenizer,
    device,
    calib_dataset_path,
    nsamples,
    sparsity_ratio,
    prune_n=0,
    prune_m=0,
    use_variant=False,
):
    """
    Apply Wanda pruning to a model.

    Wanda (Pruning by Weights AND activations) removes weights based on the
    product of weight magnitude and activation norms.

    Args:
        model: Model to prune
        tokenizer: Tokenizer for the model
        device: Device to use (cuda/cpu)
        calib_dataset_path: Path to calibration dataset JSON file
        nsamples: Number of calibration samples
        sparsity_ratio: Target sparsity ratio (0-1)
        prune_n: N for N:M structured sparsity (0 for unstructured)
        prune_m: M for N:M structured sparsity
        use_variant: Whether to use Wanda variant with adaptive threshold
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("Loading calibration data...")
    # Use a small sequence length for calibration to avoid OOM
    # 128 is commonly used in pruning papers and is sufficient for calibration
    calib_seqlen = min(model.seqlen, 128)
    dataloader = load_calibration_dataset(calib_dataset_path, nsamples, calib_seqlen, tokenizer)
    print("Dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, nsamples
        )

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # Handle multi-GPU case - determine target device for this layer
        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
        else:
            dev = device

        # Move tensors to target device if needed
        if inps.device != dev:
            inps = inps.to(dev)
            outs = outs.to(dev)
        if attention_mask is not None and attention_mask.device != dev:
            attention_mask = attention_mask.to(dev)
        if position_ids is not None and position_ids.device != dev:
            position_ids = position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0), 
                    attention_mask=attention_mask, 
                    position_ids=position_ids
                )[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"Pruning layer {i} name {name}")
            # Wanda metric: weight magnitude * sqrt(activation norm)
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = torch.zeros_like(W_metric) == 1  # Initialize mask to all False

            if prune_n != 0:
                # Structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    # Wanda variant with adaptive threshold
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"Alpha found {alpha}, sparsity {cur_sparsity:.6f}")
                else:
                    # Standard unstructured pruning
                    indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  # Set pruned weights to zero

        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
