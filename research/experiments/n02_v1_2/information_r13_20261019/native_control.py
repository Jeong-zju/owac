"""Independent frozen native RGB vision positive control, never a RAW teacher."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers.models.auto import CONFIG_MAPPING

from owac.evaluation.n02_full.information_probe import fit_readout, metrics, split_indices

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
OUT = ROOT / "outputs/n02-v12/information-r13"
EVIDENCE = ROOT / "research/experiments/n02_v1_2/information_r13_20261019"


def main():
    from starVLA.model.framework.VLM4A.PI0 import PI0
    from starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip import (
        SiglipVisionModel,
    )
    from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIMultiModalProjector

    config = CONFIG_MAPPING["paligemma"]()
    config.vision_config.intermediate_size = 4304
    config.vision_config.projection_dim = 2048
    config.vision_config.projector_hidden_act = "gelu_fast"
    vision = SiglipVisionModel(config.vision_config).eval().requires_grad_(False)
    projector = OpenPIMultiModalProjector(config).eval().requires_grad_(False)
    path = Path("/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        for module, prefix in (
            (vision, "model.paligemma_with_expert.paligemma.model.vision_tower."),
            (projector, "model.paligemma_with_expert.paligemma.model.multi_modal_projector."),
        ):
            module.load_state_dict(
                {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)},
                strict=True,
            )
    vision = vision.bfloat16().cuda()
    projector = projector.bfloat16().cuda()
    vision.vision_model.config._attn_implementation = "eager"
    x, ys, records = [], [], []
    processor = SimpleNamespace(model_name="PI05", image_resolution=(224, 224))
    with torch.inference_mode():
        for selected in json.loads((OUT / "selection.json").read_text()):
            scene = selected["scene"]
            with np.load(OUT / "snapshots" / f"scene_{scene:03d}.npz", allow_pickle=False) as a:
                rgb, y = a["rgb"], a["y"]
            for t in range(len(y)):
                images = torch.stack([PI0._image_to_tensor(processor, v) for v in rgb[t]])
                tokens = projector(vision(images.cuda().bfloat16()).last_hidden_state)
                feature = tokens.reshape(3, 16, 16, 2048).permute(0, 3, 1, 2)
                x.append(F.adaptive_avg_pool2d(feature.float(), (8, 8)).cpu().numpy())
                ys.append(y[t])
                records.append(
                    dict(group=selected["path"], split=selected["split"], scene=scene, t=t)
                )
            print(json.dumps(dict(native_scene=scene)), flush=True)
    del vision, projector
    torch.cuda.empty_cache()
    x, y = np.asarray(x), np.asarray(ys)
    ix = split_indices(records)
    rows = []
    preds = {}
    for nonlinear in (False, True):
        for seed in (13019, 13020, 13021):
            info, pred = fit_readout(x, y, records, seed=seed, nonlinear=nonlinear, device="cuda")
            head = "mlp" if nonlinear else "linear_bottleneck"
            rows.append(
                dict(
                    head=head,
                    **info,
                    metrics={s: metrics(pred[i], y[i], "controlled") for s, i in ix.items()},
                )
            )
            preds[f"{head}-{seed}"] = pred
    np.savez_compressed(
        OUT / "native_control_predictions.npz", target=y, records=json.dumps(records), **preds
    )
    report = dict(
        policy_updates=0,
        closed_loops=0,
        scope="independent RGB reference; no RAW teacher",
        weights=str(path),
        vision_load="strict complete vision tower + multimodal projector",
        preprocessing="actual PI0._image_to_tensor, letterbox224 and [-1,1]",
        rows=rows,
    )
    (EVIDENCE / "native_control_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(completed="native control")), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
