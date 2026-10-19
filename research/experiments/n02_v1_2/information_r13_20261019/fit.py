"""Train small isolated diagnostic heads, preserving test predictions and all seeds."""

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_full.information_probe import fit_readout, metrics, split_indices

ROOT = Path("/workspace/owac/worktrees/n02-measurement-completion")
OUT = ROOT / "outputs/n02-v12/information-r13"
EVIDENCE = ROOT / "research/experiments/n02_v1_2/information_r13_20261019"


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def wait_cache(path):
    deadline = time.monotonic() + 3600
    while True:
        try:
            return np.load(path, allow_pickle=False)
        except (FileNotFoundError, zipfile.BadZipFile, EOFError):
            if time.monotonic() > deadline:
                raise
            time.sleep(2)


def main(args):
    root = OUT / args.arm
    (root / "heads").mkdir(exist_ok=False)
    all_results = []
    predictions = {}
    for panel in ("controlled", "cached"):
        with wait_cache(root / (panel + ".npz")) as a:
            records = json.loads(str(a["records"].item()))
            y = a["y"]
            indices = split_indices(records)
            names = [n for n in a.files if n not in ("records", "y")]
        mean = np.broadcast_to(y[indices["train"]].mean(0), y.shape)
        all_results.append(
            dict(
                panel=panel,
                stage="training_mean",
                head="constant",
                seed=None,
                metrics={split: metrics(mean[ix], y[ix], panel) for split, ix in indices.items()},
            )
        )
        predictions[panel + "_target"] = y
        for name in names:
            with wait_cache(root / (panel + ".npz")) as a:
                x = a[name]
            for nonlinear in (False, True) if name not in ("phase", "proprio") else (True,):
                for seed in (13019, 13020, 13021):
                    head = "mlp" if nonlinear else "linear_bottleneck"
                    ident = f"{panel}-{name}-{head}-{seed}"
                    info, pred = fit_readout(
                        x,
                        y,
                        records,
                        seed=seed,
                        nonlinear=nonlinear,
                        updates=600,
                        device="cuda",
                        checkpoint_path=root / "heads" / (ident + ".pt"),
                    )
                    row = dict(
                        panel=panel,
                        stage=name,
                        head=head,
                        **info,
                        metrics={s: metrics(pred[ix], y[ix], panel) for s, ix in indices.items()},
                    )
                    all_results.append(row)
                    predictions[ident] = pred
                    print(
                        json.dumps(
                            dict(
                                panel=panel,
                                stage=name,
                                head=head,
                                seed=seed,
                                selected_step=info["selected_step"],
                                test=row["metrics"]["test"],
                            )
                        ),
                        flush=True,
                    )
                    write(
                        EVIDENCE / (args.arm + "_progress.json"),
                        dict(rows=len(all_results), last=row),
                    )
            # All visual stages receive an independently shuffled-label MLP control.
            if name not in ("phase", "proprio"):
                info, pred = fit_readout(
                    x,
                    y,
                    records,
                    seed=13019,
                    nonlinear=True,
                    shuffle=True,
                    updates=600,
                    device="cuda",
                )
                all_results.append(
                    dict(
                        panel=panel,
                        stage=name,
                        head="shuffled_mlp",
                        **info,
                        metrics={s: metrics(pred[ix], y[ix], panel) for s, ix in indices.items()},
                    )
                )
                predictions[f"{panel}-{name}-shuffled"] = pred
            del x
        predictions[panel + "_split"] = np.array([r["split"] for r in records])
        predictions[panel + "_group"] = np.array([r["group"] for r in records])
    np.savez_compressed(root / "head_predictions.npz", **predictions)
    deadline = time.monotonic() + 3600
    while not (root / "receipt.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("frozen feature audit incomplete")
        time.sleep(2)
    report = dict(
        arm=args.arm,
        policy_updates=0,
        closed_loops=0,
        head_max_updates=600,
        selection="each small head selected by validation normalized MSE; no test selection",
        frozen=json.loads((root / "receipt.json").read_text()),
        rows=all_results,
        caveats=[
            "test independence applies to diagnostic head; parent saw some cached source episodes",
            "static controlled images are a separate diagnostic distribution",
            "poor small-readout results do not prove absence of information",
            "cached world EE positions and controlled world TCP positions are separate targets",
        ],
    )
    write(EVIDENCE / (args.arm + "_report.json"), report)
    print(json.dumps(dict(completed=args.arm, heads=len(all_results) - 2)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("r11-task850", "r12-one256"), required=True)
    torch.set_num_threads(4)
    main(p.parse_args())
