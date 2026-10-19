"""Count encoder, auxiliary heads, and parameters that actually received gradients."""

import json

from run import ARCHIVE, ROOT, save, stage_config

from owac.training.n04 import build_encoder, load_checkpoint


def main():
    rows = []
    for path in sorted(ARCHIVE.glob("s*-protocol.json")):
        protocol = json.loads(path.read_text())
        assert protocol["complete"]
        stage = protocol["stage"]
        cfg, _ = stage_config(stage)
        for name in protocol["models"]:
            for run in (0, 1, 2):
                cp = ROOT / stage / name / f"run-{run}" / "checkpoint.pt"
                encoder = build_encoder(cfg, baseline=None if name == "candidate" else name)
                payload = load_checkpoint(cp, encoder=encoder)
                count = sum(p.numel() for p in encoder.parameters())
                auxiliary = sum(v.numel() for v in payload["predictor_head"].values())
                touched = sum(
                    s["exp_avg"].numel()
                    for s in payload["optimizer"]["state"].values()
                    if "exp_avg" in s
                )
                logged = json.loads((ARCHIVE / f"{stage}-training.json").read_text())
                entry = next(r for r in logged if r["name"] == name and r["run"] == run)
                assert count + auxiliary == entry["total_parameters"]
                rows.append(
                    dict(
                        stage=stage,
                        name=name,
                        run=run,
                        encoder_with_location_probe_parameters=count,
                        auxiliary_future_head_parameters=auxiliary,
                        total_optimized_parameter_capacity=count + auxiliary,
                        parameters_ever_receiving_gradient=touched,
                    )
                )
    save(
        ARCHIVE / "resource-audit.json",
        dict(
            rows=rows,
            note=(
                "Training total_parameters includes the future head; this audit verifies it. "
                "Readout is present but not task-adapted in this experiment."
            ),
        ),
    )


if __name__ == "__main__":
    main()
