"""Intervene on only change content while holding task state and observation fixed."""

import math
from dataclasses import replace

import torch
from dynamics import sequence_for
from run import ARCHIVE, ROOT, digest, save, stage_config

from owac.training.n04 import build_context, build_encoder, load_checkpoint


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    cfg, _ = stage_config("s3b-combined")
    results = []
    for run in (0, 1, 2):
        cp = ROOT / "s3b-combined" / "candidate" / f"run-{run}" / "checkpoint.pt"
        encoder = build_encoder(cfg)
        load_checkpoint(cp, encoder=encoder)
        encoder.eval()
        _, seq = sequence_for("occluded", 44000)
        state = encoder.init_state(2)["head"]
        previous = None
        for frame in seq[:8]:
            out = encoder.step(
                frame.measured,
                build_context(frame, previous),
                state,
                saturation=frame.saturated_mask,
                measurement_valid=frame.valid_mask,
            )
            state = out.state
            previous = frame
        empty = replace(state, change=[torch.zeros_like(x) for x in state.change])
        altered = replace(state, change=[torch.full_like(x, 0.5) for x in state.change])
        frame = seq[8]
        outputs = []
        for s in (empty, altered):
            outputs.append(
                encoder.step(
                    frame.measured,
                    build_context(frame, previous),
                    s,
                    saturation=frame.saturated_mask,
                    measurement_valid=frame.valid_mask,
                )
            )
        stable_diff = max(
            float((a - b).abs().max())
            for a, b in zip(outputs[0].state.stable, outputs[1].state.stable, strict=True)
        )
        memory_diff = max(
            float((a - b).abs().max())
            for a, b in zip(outputs[0].state.change, outputs[1].state.change, strict=True)
        )
        assert stable_diff == 0 and memory_diff > 0
        decay = float(encoder.change_decay.sigmoid())
        results.append(
            dict(
                run=run,
                checkpoint_sha256=digest(cp),
                hidden_stable_max_difference=stable_diff,
                hidden_change_max_difference=memory_diff,
                decay=decay,
                nominal_half_life_frames=math.log(0.5) / math.log(decay),
                write_scale=float(encoder.change_write.tanh()),
            )
        )
    save(
        ARCHIVE / "memory-path-diagnostic.json",
        dict(
            results=results,
            note=(
                "Fixed task state and missing observation; only change differs. "
                "Task probe sees stable; action/readout adaptation was not trained here."
            ),
        ),
    )


if __name__ == "__main__":
    main()
