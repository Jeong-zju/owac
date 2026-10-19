"""Two seed jobs, paired state/RAW/native residual arms; fixed1024 updates each."""

import argparse
import fcntl
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from owac.evaluation.n02_full.action_residual import fit_action_residual
from owac.frameworks.n02_full.task_chain import SmallActionHead

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/n02-v12/residual-r16"
PARENT = ROOT / "outputs/n02-v12/learning-r15"
R15 = EVIDENCE.parent / "learning_r15_20261019"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish():
    with (OUT / "publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reports = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/report.json"))]
        write(
            OUT / "progress.json",
            dict(reports=reports, completed_runs=len(reports), planned_runs=6, audited=False),
        )


def run(seed):
    if seed not in (15019, 15020):
        raise ValueError("predeclared seeds only")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = torch.load(PARENT / "dataset.pt", weights_only=False)
    selection = json.loads((R15 / "selection.json").read_text())
    groups = selection["train_groups"]["5"]
    config = json.loads((EVIDENCE.parent / "training_r11_20261017/task_config.json").read_text())
    std = json.loads(Path(config["data"]["stats_path"]).read_text())["norm_stats"]["actions"]["std"]
    baseline_dir = PARENT / f"state-5-s{seed}"
    baseline_path = baseline_dir / "checkpoint_1024.pt"
    saved = torch.load(baseline_path, weights_only=False)
    baseline = SmallActionHead().cuda()
    baseline.load_state_dict(saved["selected"], strict=True)
    baseline.eval().requires_grad_(False)
    parent_info = dict(
        checkpoint=str(baseline_path),
        sha256=sha(baseline_path),
        selected_step=saved["selected_step"],
        selected_validation_mse=saved["selected_validation_mse"],
    )
    # Cached features have no gradients or online teacher; all arms preserve positions.
    for mode in ("state", "raw", "native"):
        name = f"{mode}-s{seed}"
        directory = OUT / name
        directory.mkdir(exist_ok=False)
        torch.cuda.reset_peak_memory_stats()
        features_path = PARENT / ("native_tokens.pt" if mode == "native" else "raw_tokens.pt")
        features = torch.load(features_path, weights_only=False)
        source = dict(
            dataset_sha256=sha(PARENT / "dataset.pt"),
            feature_sha256=sha(features_path),
            baseline=parent_info,
        )
        write(directory / "source.json", source)
        started = time.time()
        with (directory / "history.jsonl").open("x") as log:

            def callback(row, name=name, mode=mode, directory=directory):
                log.write(json.dumps(row) + "\n")
                if row["step"] % 16 == 0 or row["step"] == 1:
                    log.flush()
                    status = {k: v for k, v in row.items() if k != "candidates"}
                    status.update(
                        run_id=name,
                        modality=mode,
                        seed=seed,
                        state="running",
                        updated_at=time.time(),
                    )
                    write(directory / "status.json", status)
                    if row["step"] % 256 == 0:
                        print(json.dumps(status), flush=True)

            def checkpoint(budget, payload, directory=directory):
                path = directory / f"checkpoint_{budget:04}.pt"
                torch.save(payload, path)

            report, predictions = fit_action_residual(
                baseline,
                features,
                data["state"],
                data["actions"],
                data["mask"],
                data["records"],
                groups,
                std,
                seed=seed,
                state_only=mode == "state",
                callback=callback,
                checkpoint=checkpoint,
            )
        # Exact same frozen baseline as previous validation-selected state run.
        with np.load(baseline_dir / "predictions.npz") as old:
            for split in ("train", "val", "test"):
                np.testing.assert_array_equal(
                    predictions["baseline_" + split], old["1024_selected_" + split][0]
                )
        np.savez_compressed(directory / "predictions.npz", **predictions)
        write(directory / "sample_order.json", report.pop("order"))
        report.update(
            run_id=name,
            modality=mode,
            completed=True,
            baseline_source=parent_info,
            source=source,
            baseline_matches_r15_exact=True,
            wall_elapsed_s=time.time() - started,
            peak_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        write(directory / "report.json", report)
        write(EVIDENCE / (name + "_report.json"), report)
        write(
            directory / "status.json",
            dict(
                run_id=name,
                modality=mode,
                seed=seed,
                state="completed",
                step=1024,
                max_updates=1024,
                actual_updates=1024,
                seconds_per_update=report["elapsed_s"] / 1024,
                updated_at=time.time(),
            ),
        )
        publish()
        del features, predictions, report
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        run(args.seed)
    except Exception as exc:
        write(
            OUT / f"seed{args.seed}-error.json",
            dict(failure_type=type(exc).__name__, seed=args.seed, updated_at=time.time()),
        )
        raise
