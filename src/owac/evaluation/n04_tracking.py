"""Frozen instantaneous RAW measurement -> hold / author RKN experiment.

No recurrent N04 task state enters the measurement head. Labels are used only to
train/calibrate on training scenes and to score predictions. Final scenes are
generated after the model/configuration freeze, with a fresh seed namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..baselines.n04_tracking import RKNPositionTracker, hold_last
from ..frameworks.n04.synthetic import SceneConfig, SyntheticSequenceGenerator
from ..training.n04 import build_encoder, config_from_json, load_checkpoint

KINDS = ("object_cv", "mixed_cv", "camera_only", "stop", "reverse")
DURATIONS = (0, 2, 4, 6)
MISSING_START = 10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tensor_hash(values) -> str:
    value = hashlib.sha256()
    for tensor in values:
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def centroid(mask: torch.Tensor) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(mask.shape[-2], device=mask.device),
        torch.arange(mask.shape[-1], device=mask.device),
        indexing="ij",
    )
    mass = mask.sum((-2, -1)).clamp_min(1e-6)
    return torch.stack(((mask * x).sum((-2, -1)) / mass, (mask * y).sum((-2, -1)) / mass), -1)[:, 0]


class InstantaneousMeasurement(nn.Module):
    def __init__(self, stem: nn.Module):
        super().__init__()
        self.stem = stem.requires_grad_(False).eval()
        self.head = nn.Conv2d(stem.width, 1, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # There is no state, pose, timestamp, or frame identity in this interface.
        with torch.no_grad():
            feature = self.stem(values)
        return self.head(feature)

    @torch.no_grad()
    def positions(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self(values).sigmoid() >= 0.5
        valid = mask.flatten(1).any(-1)
        return centroid(mask.float()), valid


@dataclass
class CachedScene:
    seed: int
    kind: str
    position: torch.Tensor
    detected: torch.Tensor
    target: torch.Tensor
    camera: torch.Tensor
    raw_sha256: str


def observation_inputs(scene: CachedScene, duration: int):
    valid = scene.detected.clone()
    if duration:
        valid[MISSING_START : MISSING_START + duration] = False
    # No hidden measurement reaches a tracker, including a precomputed detector output.
    position = scene.position.masked_fill(~valid[..., None], float("nan"))
    return position, valid


def frame_sequence(seed: int, kind: str, *, batch: int = 8, length: int = 24):
    """Repaired renderer, bounded paths, random initial positions and per-sample rates."""
    cfg = SceneConfig(canvas_margin=64, noise_sigma=0.01, seed=seed)
    generator = SyntheticSequenceGenerator(cfg)
    rng = torch.Generator().manual_seed(seed + 101)
    base = 32 + (torch.rand(batch, 2, generator=rng) - 0.5) * 6
    centers = (base + cfg.canvas_margin)[:, None]

    def velocity(lo, hi):
        direction = torch.randn(batch, 2, generator=rng)
        direction /= direction.norm(dim=-1, keepdim=True)
        return direction * torch.empty(batch, 1).uniform_(lo, hi, generator=rng)

    obj_v = velocity(0.3, 0.65) if kind != "camera_only" else torch.zeros(batch, 2)
    cam_v = velocity(0.1, 0.25) if kind in ("mixed_cv", "camera_only") else torch.zeros(batch, 2)
    previous_camera = torch.zeros(batch, 2)
    for t in range(length):
        distance = min(t, 12) if kind == "stop" else t if kind != "reverse" else min(t, 24 - t)
        camera = t * cam_v
        frame = generator.frame_batch(
            camera_translation=camera,
            camera_rotation=torch.zeros(batch),
            object_translation=distance * obj_v,
            illumination=torch.ones(batch),
            base_object_centers=centers,
            base_camera_offset=torch.zeros(batch, 2),
            camera_step=camera - previous_camera,
            generator=rng,
        )
        # Geometry and clipping must not silently change the intended trajectory.
        assert (frame.object_mask.sum((1, 2, 3)) > 100).all(), "target left the image"
        yield frame
        previous_camera = camera


@torch.no_grad()
def cache_scenes(measurement, seeds: list[int], *, device):
    scenes = []
    for i, seed in enumerate(seeds):
        kind = KINDS[i % len(KINDS)]
        positions, detected, targets, cameras = [], [], [], []
        raw_hash = hashlib.sha256()
        for frame in frame_sequence(seed, kind):
            raw_hash.update(frame.measured.contiguous().numpy().tobytes())
            position, available = measurement.positions(frame.measured.to(device))
            camera = frame.proprio[:, :2]
            positions.append(position.cpu() + camera)
            detected.append(available.cpu())
            targets.append(centroid(frame.object_mask) + camera)
            cameras.append(camera)
        scenes.append(
            CachedScene(
                seed,
                kind,
                torch.stack(positions),
                torch.stack(detected),
                torch.stack(targets),
                torch.stack(cameras),
                raw_hash.hexdigest(),
            )
        )
    return scenes


def train_measurement(stem, *, device, steps=200):
    torch.manual_seed(600000)
    model = InstantaneousMeasurement(stem).to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.01)
    history = []
    for step in range(steps):
        seed = 600000 + step
        renderer = SyntheticSequenceGenerator(
            SceneConfig(canvas_margin=32, noise_sigma=0.01, seed=seed)
        )
        rng = torch.Generator().manual_seed(seed + 10000)
        # Train solely on independent instantaneous images, including varied positions.
        centers = 32 + torch.empty(16, 1, 2).uniform_(8, 56, generator=rng)
        frame = renderer.frame_batch(
            camera_translation=torch.zeros(16, 2),
            camera_rotation=torch.zeros(16),
            object_translation=torch.zeros(16, 2),
            illumination=torch.ones(16),
            base_object_centers=centers,
            base_camera_offset=torch.zeros(16, 2),
            camera_step=torch.zeros(16, 2),
            generator=rng,
        )
        logits = model(frame.measured.to(device))
        loss = F.binary_cross_entropy_with_logits(
            logits, frame.object_mask.to(device), pos_weight=logits.new_tensor(8.0)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model.eval().requires_grad_(False), history


@torch.no_grad()
def evaluate_tracker(tracker, scenes, *, method, duration):
    rows, outputs = [], []
    for scene in scenes:
        position, available = observation_inputs(scene, duration)
        started = time.perf_counter()
        result = tracker(position, available)
        elapsed = (time.perf_counter() - started) * 1000 / len(position)
        error = (result.position - scene.target).norm(dim=-1)
        score = error[2:]
        row = dict(
            method=method,
            scene_seed=scene.seed,
            kind=scene.kind,
            missing_frames=duration,
            mean_error_px=float(score.mean()),
            visible_error_px=float(error[2:][available[2:]].mean()),
            hidden_error_px=float(error[MISSING_START : MISSING_START + duration].mean())
            if duration
            else None,
            recovery_error_px=float(
                error[MISSING_START + duration : MISSING_START + duration + 4].mean()
            )
            if duration
            else None,
            after_event_error_px=float(error[12:].mean())
            if scene.kind in ("stop", "reverse")
            else None,
            detector_missing=int((~scene.detected).sum()),
            tracker_ms_per_step=elapsed,
            input_sha256=tensor_hash((position, available)),
        )
        if result.variance is not None:
            assert torch.isfinite(result.variance).all() and (result.variance > 0).all()
        assert torch.isfinite(result.position).all()
        rows.append(row)
        outputs.append(
            dict(
                seed=scene.seed,
                kind=scene.kind,
                duration=duration,
                position=result.position,
                variance=result.variance,
                error=error,
            )
        )
    return rows, outputs


def selection_score(rows):
    primary = [r for r in rows if r["kind"] in ("object_cv", "mixed_cv")]
    hidden = statistics.fmean(r["hidden_error_px"] for r in primary if r["missing_frames"])
    continuous = statistics.fmean(r["mean_error_px"] for r in primary if not r["missing_frames"])
    return hidden + 0.25 * continuous


def run_all(tracker, scenes, name):
    rows, outputs = [], []
    for duration in DURATIONS:
        part, predictions = evaluate_tracker(tracker, scenes, method=name, duration=duration)
        rows.extend(part)
        outputs.extend(predictions)
    return rows, outputs


def summarize(rows):
    result = {}
    for method in sorted({r["method"] for r in rows}):
        primary = [
            r for r in rows if r["method"] == method and r["kind"] in ("object_cv", "mixed_cv")
        ]
        continuous = [r["mean_error_px"] for r in primary if not r["missing_frames"]]
        hidden = [r["hidden_error_px"] for r in primary if r["missing_frames"]]
        result[method] = dict(
            continuous_px=statistics.fmean(continuous), hidden_px=statistics.fmean(hidden)
        )
        for kind in KINDS:
            part = [r for r in rows if r["method"] == method and r["kind"] == kind]
            result[method][kind] = dict(
                continuous_px=statistics.fmean(
                    r["mean_error_px"] for r in part if not r["missing_frames"]
                ),
                hidden_px=statistics.fmean(
                    r["hidden_error_px"] for r in part if r["missing_frames"]
                ),
            )
    reference = result["unfiltered_hold"]
    for item in result.values():
        item["hidden_improvement_fraction"] = 1 - item["hidden_px"] / reference["hidden_px"]
        item["benefit_gate"] = bool(
            item["hidden_improvement_fraction"] >= 0.10
            and reference["hidden_px"] - item["hidden_px"] >= 0.05
            and item["continuous_px"]
            <= reference["continuous_px"] + max(0.05, 0.10 * reference["continuous_px"])
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rkn-steps", type=int, default=1200)
    args = parser.parse_args()
    root = args.output_dir
    if root.exists():
        raise FileExistsError("Use a fresh output directory; never overwrite frozen evidence")
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_checkpoint = Path(
        "outputs/n04-staged-rebuild-20261013/s3b-combined/candidate/run-0/checkpoint.pt"
    )
    cfg, _ = config_from_json(Path("configs/encoders/n04/tiny-staged-research.json"))
    encoder = build_encoder(cfg)
    load_checkpoint(source_checkpoint, encoder=encoder)
    protocol = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_checkpoint=str(source_checkpoint),
        source_checkpoint_sha256=digest(source_checkpoint),
        head_steps=200,
        head_seed_namespace=[600000, 600199],
        train_seeds=list(range(610000, 610064)),
        development_seeds=list(range(620000, 620030)),
        reserved_test_seeds=list(range(630000, 630030)),
        rkn_steps=args.rkn_steps,
        rkn_initializations=[0, 1, 2],
        missing_start=MISSING_START,
        durations=DURATIONS,
        factor="temporal state only; identical frozen per-frame stem/head/measurement cache",
        camera_context="same online synthetic proprio translation for all, zero camera rotation",
        gate="hidden improvement >=10% and >=0.05px; continuous regression <=max(0.05px,10%)",
        information="RAW current image only in measurement; no GT or future image in tracker",
        unfiltered_missing=(
            "hold last world-coordinate observation, using same camera pose conversion"
        ),
        torch=torch.__version__,
        cpu_threads=1,
        measurement_device=device,
        tracking_device="cpu",
        complete=False,
    )
    files = [
        Path(__file__),
        Path("src/owac/baselines/n04_tracking.py"),
        *Path("src/owac/baselines/_vendor/rkn").glob("*.py"),
        Path("src/owac/frameworks/n04/synthetic.py"),
    ]
    protocol["source_sha256"] = {str(p): digest(p) for p in files}
    save_json(root / "protocol.json", protocol)
    measurement, history = train_measurement(encoder.stem, device=device)
    torch.save(measurement.state_dict(), root / "measurement.pt")
    frozen_sensor_hash = tensor_hash(measurement.state_dict().values())
    train = cache_scenes(measurement, protocol["train_seeds"], device=device)
    dev = cache_scenes(measurement, protocol["development_seeds"], device=device)
    torch.save({"train": train, "development": dev}, root / "development-cache.pt")
    residual = torch.cat([(s.position - s.target)[s.detected] for s in train])
    variance = residual.square().mean(0).clamp_min(0.001)
    protocol.update(
        measurement_state_sha256=frozen_sensor_hash,
        measurement_variance_px2=variance.tolist(),
        head_loss_first=history[0],
        head_loss_last=history[-1],
    )
    save_json(root / "protocol.json", protocol)
    print("measurement frozen", variance.tolist(), history[-1], flush=True)
    training = []
    for seed in protocol["rkn_initializations"]:
        torch.manual_seed(seed)
        model = RKNPositionTracker(variance)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        rng = torch.Generator().manual_seed(640000 + seed)
        best = float("inf")
        for step in range(args.rkn_steps):
            index = int(torch.randint(len(train), (), generator=rng))
            scene = train[index]
            duration = DURATIONS[int(torch.randint(len(DURATIONS), (), generator=rng))]
            z, valid = observation_inputs(scene, duration)
            result = model(z, valid)
            # Supervised labels appear only in the loss, including absent-frame targets.
            loss = ((result.position[2:] - scene.target[2:]) / 32).square().mean()
            loss = loss + 0.1 * ((result.prior[2:] - scene.target[2:]) / 32).square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % 200 == 0 or step + 1 == args.rkn_steps:
                rows, _ = run_all(model.eval(), dev, f"rkn-{seed}")
                score = selection_score(rows)
                training.append(
                    dict(seed=seed, step=step + 1, loss=float(loss.detach()), dev_score=score)
                )
                if score < best:
                    best = score
                    torch.save(model.state_dict(), root / f"rkn-{seed}.pt")
                print("RKN", seed, step + 1, float(loss.detach()), score, flush=True)
    save_json(root / "training.json", training)
    # Freeze ALL selection decisions before generating or reading new final scenes.
    freeze = dict(
        protocol,
        rkn_checkpoint_sha256={str(seed): digest(root / f"rkn-{seed}.pt") for seed in range(3)},
    )
    save_json(root / "freeze.json", freeze)
    test = cache_scenes(measurement, protocol["reserved_test_seeds"], device=device)
    assert frozen_sensor_hash == tensor_hash(measurement.state_dict().values())
    torch.save(test, root / "final-cache.pt")
    trackers = {
        "unfiltered_hold": hold_last,
    }
    for seed in range(3):
        model = RKNPositionTracker(variance)
        model.load_state_dict(torch.load(root / f"rkn-{seed}.pt", weights_only=True))
        trackers[f"rkn-{seed}"] = model.eval()
    all_rows, predictions = [], {}
    for name, tracker in trackers.items():
        rows, outputs = run_all(tracker, test, name)
        all_rows.extend(rows)
        predictions[name] = outputs
    save_json(root / "final-matrix.json", all_rows)
    torch.save(predictions, root / "predictions.pt")
    summary = summarize(all_rows)
    save_json(root / "summary.json", summary)
    protocol.update(
        complete=True,
        dynamic_benefit=any(
            v["benefit_gate"] for k, v in summary.items() if k != "unfiltered_hold"
        ),
    )
    assert all(digest(Path(p)) == h for p, h in protocol["source_sha256"].items())
    save_json(root / "protocol.json", protocol)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
