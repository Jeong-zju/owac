"""Re-render true saved-run attention after fixing premature uint8 conversion; no training."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from owac.evaluation.n02_full import v12 as ev
from owac.evaluation.n02_full import v12_viz as viz
from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs
from owac.training.n02_v12 import N02V12Trainer

torch.set_num_threads(4)
base = Path("outputs/n02-v12/P2/independent-audit-20261016")
receipts = []
for platform in ("robotwin", "libero"):
    settings = ev.resolve_settings(
        ev.build_parser().parse_args(
            ["--config", f"configs/encoders/n02_v12/raw_n02_{platform}.json", "--no-publish-status"]
        )
    )
    model = ev.build_model(settings, device="cuda")
    if platform == "robotwin":
        ev.load_checkpoint_into(model, base / platform / "checkpoint_0000002.pt")
    source = ev.build_source(settings, "dev")
    indices = ev.anchor_indices(source, 2) if platform == "robotwin" else [source.anchor_count // 2]
    directory = (
        base / platform / "preview"
        if platform == "robotwin"
        else base / platform / "sampling-probe" / "preview"
    )
    with torch.no_grad():
        for index in indices:
            batch = source.window(index, index + 1)
            replay = replay_inputs(model, window_inputs(batch))
            capture = viz._anchor_capture(model, batch, device="cuda", replay=replay)
            capture["model"] = model
            attention = viz._attention_maps(capture, settings)
            for camera in settings.camera_ids():
                frame = ev.row_observation(batch.observation, len(batch.frame_index) - 1).frames[
                    camera
                ]
                white, gamma, gains = viz._declared_display(
                    viz._camera_grid(frame)[2], frame, platform
                )
                filename = (
                    f"02_attention_{index}_{camera}.svg"
                    if platform == "robotwin"
                    else f"attention_{camera}.svg"
                )
                svg = viz.figure_attention(
                    frame,
                    attention,
                    camera_id=camera,
                    labels=viz._labels(
                        settings,
                        source.anchor(index).as_dict(),
                        camera,
                        int(batch.frame_index[-1]),
                        3,
                        2 if platform == "robotwin" else 0,
                    ),
                    title="real attention; float-preserving resize; no training",
                    white_level=white,
                )
                (directory / filename).write_bytes(svg)
                receipts.append(str(directory / filename))
        # The per-training-preview sample uses a calibration window for RoboTwin.
        if platform == "robotwin":
            source = ev.build_source(settings, "calibration")
            batch = source.window(source.anchor_count // 2, source.anchor_count // 2 + 1)
            output = replay_inputs(model, window_inputs(batch))["rows"][-1]
            trainer = N02V12Trainer.__new__(N02V12Trainer)
            trainer.plan = SimpleNamespace(previews=1)
            trainer.model, trainer.run_dir, trainer.current_step = model, base / platform, 2
            receipts.append(trainer.write_previews(batch, output))
    del source, model
    torch.cuda.empty_cache()
(base / "attention_render_repair.json").write_text(
    json.dumps(
        {
            "optimizer_updates": 0,
            "reason": "retain floating attention magnitudes until heat normalisation",
            "files": receipts,
        },
        indent=2,
    )
    + "\n"
)
print("attention regenerated without optimizer updates", flush=True)
