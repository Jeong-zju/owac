"""Fixed-sample diagnosis only: compare fresh vs inherited decoder on the corrected objective."""

import json
from pathlib import Path

import torch

from owac.evaluation.n02_full import v12 as ev
from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs

torch.set_num_threads(4)
settings = ev.resolve_settings(
    ev.build_parser().parse_args(
        [
            "--config",
            "configs/encoders/n02_v12/raw_n02_robotwin.json",
            "--no-publish-status",
        ]
    )
)
source = ev.build_source(settings, "train")
index = 42584
batch = source.window(index, index + 1)
model = ev.build_model(settings, device="cuda")
base = Path("outputs/n02-v12/P2/independent-audit-20261016")
result = {
    "anchor_index": index,
    "identity": source.anchor(index).as_dict(),
    "optimizer_updates": 0,
    "purpose": "scale diagnosis, no capability inference",
}


def measure():
    with torch.no_grad():
        causal = replay_inputs(model, window_inputs(batch))["rows"][-1].causal
        energy, _ = model.inference.measurement.objective(
            mean=causal["mean"],
            variance=causal["variance"],
            masks=causal["valid"],
            target=causal["queries"].target,
        )
        mass = causal["valid"].sum(dim=-1).clamp_min(1)
        report = {"mean_causal_nll_per_scalar": float((energy / mass).mean())}
        for name, tensor in (
            ("variance", causal["variance"]),
            ("target", causal["queries"].target),
            ("prediction", causal["mean"]),
            ("abs_standardized_residual", causal["residuals"].abs()),
        ):
            report[name] = [
                float(x)
                for x in torch.quantile(
                    tensor.float().flatten(), torch.tensor([0.0, 0.5, 0.95, 1.0], device="cuda")
                )
            ]
        return report


result["fresh_full_frontend"] = measure()
legacy = Path("outputs/n02-v12/P2/raw-n02-robotwin-s1103/checkpoint_0001000.pt")
payload = torch.load(legacy, map_location="cpu", weights_only=False)
model.load_state_dict(payload["model"])
del payload
result["legacy_full_frontend"] = measure()
# Isolate decoder initialization with the inherited encoder, proposals and readout unchanged.
torch.manual_seed(settings.seed)
fresh = ev.build_model(settings, device="cpu")
model.inference.decoder.load_state_dict(fresh.inference.decoder.state_dict())
del fresh
result["legacy_with_fresh_decoder"] = measure()
(base / "causal_scale_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
