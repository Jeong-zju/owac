import json
from pathlib import Path

import numpy as np
import torch

from owac.frameworks.n04.contracts import INPUT_SCALES
from owac.frameworks.n04.correspondence import (
    RaftCorrespondence,
    RaftCorrespondenceLoss,
    RawScale,
    bidirectional_correspondence,
)

out = Path("outputs/n04-v11-s2-repair-20261016")
receipts = []
for platform, scale_name, white in [
    ("libero", "libero_uint8", 1.0),
    ("robotwin", "robotwin_float_linear", 3.5),
]:
    torch.manual_seed(101)
    source = Path(f"outputs/n04-v11-s2-independent-20261016/{platform}-dev-twoframes.npz")
    with np.load(source) as z:
        camera = next(k for k in z.files if k not in ("metadata_json", "proprio"))
        frames = torch.as_tensor(z[camera].copy(), device="cuda", dtype=torch.float32)
    scale = RawScale.from_input_scale(INPUT_SCALES[scale_name], saturation_level=white)
    model = RaftCorrespondence(scale=scale, iters=12).cuda().train()
    objective = RaftCorrespondenceLoss(scale=scale)
    objective.set_step(1000)
    versions = [p._version for p in model.parameters()]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            teacher = bidirectional_correspondence(model, frames[:1], frames[1:])
        loss = objective.self_supervised(
            model, frames[:1], frames[1:], teacher.flow_px, teacher.backward_flow_px
        )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert float(loss.detach()) > 0 and grads and all(bool(torch.isfinite(g).all()) for g in grads)
    gradient_sum = sum(float(g.float().abs().sum()) for g in grads)
    assert gradient_sum > 0 and versions == [p._version for p in model.parameters()]
    record = {
        "platform": platform,
        "camera": camera,
        "input": str(source),
        "input_hw": list(frames.shape[-2:]),
        "selfsup_only_loss": float(loss.detach()),
        "finite_gradient_tensors": len(grads),
        "gradient_abs_sum": gradient_sum,
        "teacher_requires_grad": teacher.flow_px.requires_grad,
        "optimizer_updates": 0,
        "model_initialization": "random",
        "schedule_index_for_ramp": 1000,
        "parameter_versions_unchanged": True,
        "device": torch.cuda.get_device_name(),
    }
    receipts.append(record)
    print(json.dumps(record), flush=True)
    del model, grads, teacher, loss
    torch.cuda.empty_cache()
(out / "selfsup-real-input.json").write_text(json.dumps(receipts, indent=2) + "\n")
