"""LoRA weight parametrization for the π0.5 feature bridge.

Hu et al. (2021), microsoft/LoRA @ c4593f060e6a368d7bb5af5273b8e42810cdef90.
The rank/alpha scaling and zero-B initialization follow loralib (MIT).
Parametrizing weight also covers the pinned backend's direct weight accesses;
replacing only nn.Linear.forward would leave those paths unadapted.
"""

import math

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankWeight(nn.Module):
    def __init__(self, weight, *, rank=8, alpha=16):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.scale = float(alpha) / rank
        self.A = nn.Parameter(torch.empty(rank, weight.shape[1], device=weight.device))
        self.B = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, original):
        delta = (self.B @ self.A).to(original.dtype)
        return original + self.scale * delta


def install_low_rank(base, settings):
    if not settings:
        return {}, {"kind": "none", "trainable_parameters": 0}
    if settings.get("kind") != "lora":
        raise ValueError("only an explicitly declared LoRA backend adapter is supported")
    if any(parameter.requires_grad for parameter in base.parameters()):
        raise ValueError("install LoRA only after freezing the original backend")
    targets = tuple(settings.get("targets", ("q_proj", "v_proj")))
    excluded_prefixes = tuple(settings.get("exclude_prefixes", ()))
    if any(not isinstance(prefix, str) or not prefix for prefix in excluded_prefixes):
        raise ValueError("LoRA exclusion prefixes must be nonempty strings")
    rank, alpha = int(settings.get("rank", 8)), float(settings.get("alpha", 16))
    adapters = {}
    for name, module in list(base.named_modules()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in excluded_prefixes):
            continue
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            if parametrize.is_parametrized(module, "weight"):
                raise ValueError(f"backend weight is already parametrized: {name}")
            adapter = LowRankWeight(module.weight, rank=rank, alpha=alpha)
            parametrize.register_parametrization(module, "weight", adapter)
            adapters[name] = adapter
    if not adapters:
        raise ValueError("declared LoRA targets matched no backend modules")
    expected = {id(p) for adapter in adapters.values() for p in adapter.parameters()}
    actual = {id(p) for p in base.parameters() if p.requires_grad}
    if expected != actual:
        raise RuntimeError("undeclared backend parameters became trainable")
    return adapters, {
        "kind": "lora",
        "rank": rank,
        "alpha": alpha,
        "targets": targets,
        "exclude_prefixes": excluded_prefixes,
        "modules": list(adapters),
        "trainable_parameters": sum(p.numel() for a in adapters.values() for p in a.parameters()),
        "original_backend_frozen": True,
    }


def adapter_state(adapters):
    return {
        name: {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        for name, adapter in adapters.items()
    }


def restore_adapters(adapters, saved):
    if set(adapters) != set(saved):
        raise ValueError("backend adapter checkpoint contract mismatch")
    for name, adapter in adapters.items():
        adapter.load_state_dict(saved[name], strict=True)
