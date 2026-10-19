"""Neural-only CUDA acceleration; physical observations and support remain FP32."""

from contextlib import nullcontext

import torch


def neural_autocast(module, tensor):
    dtype = getattr(module, "neural_dtype", None)
    if tensor.device.type == "cuda" and dtype is not None:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def configure_acceleration(
    model, device, precision="fp32", *, skip_empty_sources=True, pair_chunk=1
):
    device = torch.device(device)
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires a supported CUDA device")
    dtype = torch.bfloat16 if precision == "bf16" else None
    model.encoder.neural_dtype = dtype
    model.encoder.skip_empty_sources = skip_empty_sources
    model.encoder.content_frame_chunk = 4
    provider = getattr(model.encoder, "motion_provider", None)
    if provider is not None:
        provider.neural_dtype = dtype
        provider.pair_chunk = pair_chunk
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return {
        "precision": precision,
        "physical_statistics": "fp32",
        "skip_empty_sources": skip_empty_sources,
        "flow_pair_batch_size": pair_chunk,
        "content_batch_size": 4,
    }
