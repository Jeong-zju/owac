"""Evaluate registered support/noise/risk candidates against labelled real controls.

No optimizer, automatic threshold selection or unlabelled false-alarm claim. Each
label binds a real episode/camera/source id/region. Calibration and validation
regions are evaluated separately; the test region is deliberately inaccessible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from owac.evaluation.n01_v12 import _datasets
from owac.frameworks.n01_full.model import N01V12Config, build_model
from owac.frameworks.n01_full.support import SUPPORT_STATE_NAMES
from owac.raw.rawvla_sequence import collate_sequence_samples
from owac.training.n01_v12 import encode_windows_v12
from owac.training.n01_v12_no_update import forbid_optimizer_updates


def architecture(config):
    data = config.to_dict()
    data.pop("support")
    for key in ("shot", "read_variance", "noise_provenance"):
        data["observation"].pop(key)
    data["motion"].pop("max_cycle_error")
    return data


def label_counts(state, camera, labels):
    buffer = state.cameras[camera]
    decisions = state.support[camera]["state"]
    result = Counter()
    for label in labels:
        expected = label["expected"]
        if expected not in ("stable", "change"):
            raise ValueError("labels must describe stable or persistent structural change")
        positions = (buffer.frame_ids[:, 0] == int(label["source_frame_id"])).nonzero().flatten()
        if len(positions) != 1 or int(positions[0]) == 0:
            raise ValueError(
                "label must identify one retained historical source, not the reference"
            )
        region = int(label["region"])
        if not 0 <= region < decisions.shape[-1]:
            raise ValueError("label region outside configured grid")
        actual = SUPPORT_STATE_NAMES[int(decisions[0, int(positions[0]) - 1, region])]
        result[f"{expected}/{actual}"] += 1
    return result


def run(args):
    controls = json.loads(Path(args.labels).read_text())
    if controls.get("schema") != "n01-labelled-support-controls-v1":
        raise ValueError("label schema must bind actual episode/source/region identities")
    results = []
    for candidate in args.candidate_config:
        config = N01V12Config.from_dict(json.loads(Path(candidate).read_text()))
        if not config.spec.support_enabled or config.support.implementation != "gls_window":
            raise ValueError("labelled support calibration requires the complete N01 arm")
        model = build_model(config, device=args.device).eval()
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            original = N01V12Config.from_dict(payload["config"])
            if architecture(original) != architecture(config):
                raise ValueError("only declared support/noise/risk candidate fields may differ")
            model.load_state_dict(payload["model"])
        elif not args.allow_untrained:
            raise ValueError("supply learned weights or explicitly request untrained diagnostics")
        datasets, totals = {}, {}
        for control in controls["controls"]:
            region = control["split"]
            if region not in ("calibration", "validation"):
                raise ValueError("test/train regions cannot select support operating points")
            if region not in datasets:
                datasets[region] = _datasets(SimpleNamespace(**vars(args)), config, region)
            sample = datasets[region][int(control["sample_index"])]
            if str(sample.meta["episode_key"]) != control["episode_key"]:
                raise ValueError("control identity differs from the current data manifest")
            camera = control["camera"]
            counts = Counter()

            def capture(name, output, camera=camera, counts=counts, control=control):
                if name == camera:
                    counts.update(label_counts(output.state, camera, control["labels"]))
                return output

            encode_windows_v12(
                model, collate_sequence_samples([sample]), args.device, transform=capture
            )
            if not counts:
                raise ValueError("labelled camera did not execute")
            totals.setdefault(f"{region}/{camera}", Counter()).update(counts)
        results.append(
            {
                "config": config.to_dict(),
                "counts": totals,
                "checkpoint": args.checkpoint,
                "candidate": candidate,
            }
        )
    receipt = {
        "schema": "n01-labelled-operating-point-comparison-v1",
        "results": results,
        "optimizer_updates": 0,
        "automatically_selected": None,
        "boundary": "empirical labelled counts per camera/split; no anytime error guarantee",
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", action="append", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--protocol")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with forbid_optimizer_updates(), torch.no_grad():
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
