"""Paired teacher/student closed-loop rollout on LIBERO task6 RAW."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.evaluation.raw_all_bench_actions import (
    _build_bridge,
    _token_view,
    deployment_state8,
    normalize_checkpoint_state,
)
from owac.evaluation.rawvla_teacher import load_pinned_postprocessor
from owac.raw.all_bench_distillation import (
    build_all_bench_student,
    build_vision_teacher,
    teacher_embed_rgb,
)
from owac.simulation.rawvla_teacher_controller import (
    INSTRUCTION,
    LIBEROEnvAdapter,
)


@dataclass(frozen=True, slots=True)
class Attempt:
    role: str
    episode_id: str
    seed: int
    array_index: int


def load_test_states(catalog: str | Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    path = Path(catalog)
    payload = json.loads(path.read_text(encoding="utf-8"))
    states_path = Path(str(payload["states_path"]))
    if not states_path.is_absolute() and not states_path.is_file():
        states_path = path.parent / states_path
    archive = np.load(states_path, allow_pickle=False)
    if isinstance(archive, np.lib.npyio.NpzFile):
        states = np.asarray(archive["states"])
        archive.close()
    else:
        states = np.asarray(archive)
    entries = [dict(item) for item in payload.get("episodes", []) if item.get("split") == "test"]
    if not entries:
        raise RuntimeError("catalog has no test episodes")
    return entries, states


def _episode_seed(entry: Mapping[str, Any]) -> int:
    return int(entry["seed"])


def paired_noise(seed: int, episode_id: str, decision_step: int) -> torch.Tensor:
    # Numeric deterministic rule; no hashing.
    numeric = sum((index + 1) * ord(char) for index, char in enumerate(str(episode_id)))
    value = (int(seed) * 1000003 + int(decision_step) * 7919 + numeric) & 0xFFFFFFFF
    return torch.randn(
        (1, 10, 32), generator=torch.Generator().manual_seed(value), dtype=torch.float32
    )


def _raw_obs(
    observation: Mapping[str, Any], cfg: Mapping[str, Any], *, seed: int, frame_index: int
) -> dict[str, np.ndarray]:
    from rawvla_bench.raw import make_rawvla_observation

    frames = observation["raw"]
    result = {
        key: make_rawvla_observation(
            frames[key],
            noise_seed=int(cfg.get("noise_seed", seed)),
            frame_index=int(frame_index),
            view_index=idx,
            representation="raw",
            sensor_saturation_ev=float(cfg.get("sensor_saturation_ev", 0.0)),
        )
        for idx, key in enumerate(("agentview", "wrist"))
    }
    result["frame_index"] = int(frame_index)
    return result


def _push_raw(ring: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    """Append one physical observation, dropping an adjacent duplicate index."""
    if ring and ring[-1].get("frame_index") == item.get("frame_index"):
        return ring[-6:]
    return (ring + [item])[-6:]


def _burst(ring: list[dict[str, Any]], key: str) -> torch.Tensor:
    hist = [item[key] for item in ring]
    hist = [hist[0]] * (6 - len(hist)) + hist
    return torch.from_numpy(np.stack(hist)).permute(0, 3, 1, 2).float() / 255.0


def _packet(
    source_id: str,
    endpoint: int,
    ring: list[dict[str, Any]],
    cams: tuple[str, ...],
    dev: torch.device,
) -> Any:
    import torch.nn.functional as F

    from owac.baselines.raw_stem_siglip import RawStemInput
    from owac.baselines.raw_temporal_siglip import (
        TemporalPacket,
        causal_frame_ids,
        causal_padding_mask,
    )

    raw_keys = ("agentview", "wrist")[: len(cams)]
    camera_bursts = []
    for key in raw_keys:
        raw = _burst(ring, key)
        camera_bursts.append(
            F.interpolate(raw, size=(224, 224), mode="bilinear", align_corners=False)
        )
    packet_frames = []
    for index in range(6):
        vals = torch.stack([camera_bursts[cam][index] for cam in range(len(cams))], dim=0)
        packet_frames.append(
            RawStemInput(
                values=vals.unsqueeze(0).to(dev),
                input_mode="rawvla_rgb3",
                camera_ids=cams,
                units="normalized_0_1",
            )
        )
    return TemporalPacket(
        frames=tuple(packet_frames),
        source_demo_id=source_id,
        endpoint=endpoint,
        frame_ids=causal_frame_ids(endpoint),
        padding_mask=causal_padding_mask(endpoint),
        reset=endpoint == 10,
        timestamps=None,
        position_source_size=256,
        position_target_size=224,
        position_backmap=True,
        preprocess_id="libero_closed_loop_fullfov_256_to_224",
    )


def _postprocess(normalized: np.ndarray, state_raw: np.ndarray, stats: Any) -> np.ndarray:
    postprocess = load_pinned_postprocessor()
    return np.asarray(
        postprocess(
            normalized[0],
            np.asarray(state_raw, dtype=np.float32).reshape(8),
            {
                "actions": {
                    "q01": np.asarray(stats.action["q01"], dtype=np.float32),
                    "q99": np.asarray(stats.action["q99"], dtype=np.float32),
                }
            },
            model_name="PI05",
            model_source="openpi",
        ),
        dtype=np.float32,
    )


def _wilson(k: int, n: int) -> list[float] | None:
    if n <= 0:
        return None
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


class _TeacherTokenController:
    def __init__(self, teacher: Any, device: torch.device) -> None:
        self.teacher = teacher
        self.device = device
        self.states = {camera: None for camera in ("agentview", "wrist")}
        self.thetas = {camera: None for camera in ("agentview", "wrist")}

    def tokens(self, ring: list[dict[str, Any]]) -> Any:
        parts = []
        for camera in ("agentview", "wrist"):
            raw = _burst(ring, camera).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.teacher.frontend.model(
                    raw, state=self.states[camera], theta_prev=self.thetas[camera]
                )
                self.states[camera] = out.state.detach()
                self.thetas[camera] = out.theta.detach()
                parts.append(
                    teacher_embed_rgb(
                        self.teacher, out.rgb.detach()[0].float().clamp(0, 1).unsqueeze(0)
                    )
                )
        return _token_view(torch.cat(parts, dim=0), ("base_0_rgb", "left_wrist_0_rgb"))


class _StudentTokenController:
    def __init__(self, student: Any, device: torch.device) -> None:
        self.student = student
        self.device = device
        self.state = None

    def tokens(self, episode_id: str, endpoint: int, ring: list[dict[str, Any]]) -> Any:
        packet = _packet(
            episode_id, endpoint, ring, ("base_0_rgb", "left_wrist_0_rgb"), self.device
        )
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, self.state = self.student(packet, self.state)
        return output.tokens


def run(
    config: Mapping[str, Any],
    catalog: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cuda:0",
    max_episodes: int = 20,
    libero_config: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    entries, states = load_test_states(catalog)
    entries = entries[: int(max_episodes)]
    assets = config["assets"]
    adapter = LIBEROEnvAdapter(
        author_root=assets["author_root"],
        libero_config=libero_config or assets.get("libero_config", ""),
    )
    bridge, stats = _build_bridge(
        {
            "assets": {
                **assets,
                "tokenizer": assets["tokenizer"],
                "statistics": assets["statistics"],
            }
        },
        device=torch.device(device),
    )
    teacher = build_vision_teacher(config, device=device)
    student = build_all_bench_student(config, device=device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    student.load_state_dict(normalize_checkpoint_state(payload), strict=True)
    del payload
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    for module in (teacher.frontend, teacher.vision_tower, teacher.projector):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    instruction = INSTRUCTION
    results: list[dict[str, Any]] = []
    ledger = out / "attempts.jsonl"

    def record(row: Mapping[str, Any]) -> None:
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(row)) + "\n")

    teacher_tokens = _TeacherTokenController(teacher, torch.device(device))
    student_tokens = _StudentTokenController(student, torch.device(device))
    pairs = [(role, entry) for entry in entries for role in ("teacher", "student")]
    for role, entry in pairs:
        teacher_tokens.states = {camera: None for camera in ("agentview", "wrist")}
        teacher_tokens.thetas = {camera: None for camera in ("agentview", "wrist")}
        student_tokens.state = None
        episode_id = str(entry["id"])
        seed = int(entry["seed"])
        start = time.monotonic()
        record({"role": role, "episode_id": episode_id, "status": "started", "seed": seed})
        env_obj = adapter.bind(entry, states)
        video_path = out / f"{role}_{episode_id}.mp4"
        import imageio.v2 as imageio

        media = imageio.get_writer(str(video_path), fps=10, codec="libx264")
        ring: list[dict[str, Any]] = []
        plan = None
        steps = 0
        replans = 0
        normalized_rows, physical_rows, executed_rows, state_rows, state_raw_rows, noise_rows = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        decision = 0
        try:
            observation = env_obj.reset()
            lighting = getattr(adapter, "lighting", None) or {}
            native = observation.get("native") or {}
            first_frame = native.get("agentview_image")
            if first_frame is not None:
                media.append_data(np.asarray(first_frame, dtype=np.uint8)[::-1, ::-1])
            ring = _push_raw(ring, _raw_obs(observation, lighting, seed=seed, frame_index=0))
            for wait_index in range(10):
                observation, _, _, _ = env_obj.step([0.0] * 6 + [-1.0])
                ring = _push_raw(
                    ring, _raw_obs(observation, lighting, seed=seed, frame_index=wait_index + 1)
                )
                native = observation.get("native") or {}
                frame = native.get("agentview_image")
                if frame is not None:
                    media.append_data(np.asarray(frame, dtype=np.uint8)[::-1, ::-1])
            first_raw_burst = None
            while steps < 300:
                if steps > 0:
                    ring = _push_raw(
                        ring, _raw_obs(observation, lighting, seed=seed, frame_index=10 + steps)
                    )
                if steps % 5 == 0:
                    state_raw = np.asarray(observation["robot_state"], dtype=np.float32)
                    state8 = deployment_state8(state_raw, stats)
                    noise = paired_noise(seed, episode_id, decision).to(device)
                    if first_raw_burst is None:
                        first_raw_burst = np.stack(
                            [np.stack([item[k] for k in ("agentview", "wrist")]) for item in ring]
                        )
                    if role == "teacher":
                        tokens = teacher_tokens.tokens(ring)
                    else:
                        tokens = student_tokens.tokens(episode_id, 10 + steps, ring)
                    from owac.backends.rawvla_language import prepare_language_state

                    lang_tokens, lang_masks, state32 = prepare_language_state(
                        bridge, [instruction], state8[None], device=torch.device(device)
                    )
                    prediction = bridge.predict_action_from_visual_tokens(
                        tokens, lang_tokens, lang_masks, state32, noise=noise, num_steps=10
                    )
                    normalized = np.asarray(prediction["normalized_actions"], dtype=np.float32)
                    physical = _postprocess(normalized, state_raw, stats)
                    plan = physical
                    normalized_rows.append(normalized[0])
                    physical_rows.append(physical)
                    state_rows.append(state8)
                    state_raw_rows.append(state_raw)
                    noise_rows.append(noise.cpu().numpy()[0])
                    replans += 1
                    decision += 1
                action = plan[steps % 5]
                observation, _, done, _ = env_obj.step(action)
                executed_rows.append(np.asarray(action, dtype=np.float32))
                steps += 1
                native = observation.get("native") or {}
                frame = native.get("agentview_image")
                if frame is not None:
                    media.append_data(np.asarray(frame, dtype=np.uint8)[::-1, ::-1])
                if done or env_obj.check_success():
                    break
            success = bool(env_obj.check_success())
            first_raw = (
                first_raw_burst
                if first_raw_burst is not None
                else np.zeros((6, 2, 256, 256, 3), np.uint8)
            )
            np.savez_compressed(
                out / f"{role}_{episode_id}.npz",
                normalized=np.asarray(normalized_rows),
                physical=np.asarray(physical_rows),
                executed=np.asarray(executed_rows),
                state=np.asarray(state_rows),
                state_raw=np.asarray(state_raw_rows),
                noise=np.asarray(noise_rows),
                first_raw=first_raw,
                initial_state=np.asarray(states[int(entry["array_index"])]),
                episode_id=np.asarray([episode_id]),
                seed=np.asarray([seed]),
                success=np.asarray([success]),
                steps=np.asarray([steps]),
            )
            result = {
                "role": role,
                "episode_id": episode_id,
                "seed": seed,
                "success": success,
                "steps": steps,
                "replans": replans,
                "video": str(video_path),
                "seconds": time.monotonic() - start,
            }
            results.append(result)
            record({**result, "status": "completed"})
        except Exception as exc:
            record({"role": role, "episode_id": episode_id, "status": "error", "error": repr(exc)})
            raise
        finally:
            media.close()
            env_obj.close()
    summary = {}
    for role in ("teacher", "student"):
        rows = [r for r in results if r["role"] == role]
        success = sum(bool(r["success"]) for r in rows)
        summary[role] = {
            "success": success,
            "attempted": len(rows),
            "wilson": _wilson(success, len(rows)),
        }
    teacher_map = {r["episode_id"]: r for r in results if r["role"] == "teacher"}
    student_map = {r["episode_id"]: r for r in results if r["role"] == "student"}
    summary["paired"] = {
        "regressions": [
            e
            for e in teacher_map
            if teacher_map[e]["success"] and not student_map.get(e, {}).get("success")
        ],
        "recoveries": [
            e
            for e in student_map
            if student_map[e]["success"] and not teacher_map.get(e, {}).get("success")
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
