"""Closed-loop teacher/reference evaluation driver contract.

Importing this module does not import LIBERO or start a simulator.  Actual
execution requires an explicit environment adapter and a PhaseBudget lease; the
driver records attempts before reset and preserves every failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROLES = ("default_reference", "learned_raw_teacher")
SETUP_WAIT_STEPS = 10
MAX_VALIDATION = 10
MAX_TEST = 20


class TeacherEvalBlocked(RuntimeError):
    pass


def load_initial_states(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise TeacherEvalBlocked(f"initial state manifest absent: {source}")
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise TeacherEvalBlocked("initial state manifest hash differs")
    payload = json.loads(source.read_text())
    if payload.get("schema") != "owac.raw-teacher.initial-states.v1":
        raise TeacherEvalBlocked("initial state schema differs")
    states_path = Path(str(payload["states_path"]))
    if not states_path.is_file():
        raise TeacherEvalBlocked(f"initial state array absent: {states_path}")
    if hashlib.sha256(states_path.read_bytes()).hexdigest() != payload.get("states_sha256"):
        raise TeacherEvalBlocked("initial state array hash differs")
    episodes = payload.get("episodes", [])
    validation = [item for item in episodes if item.get("split") == "validation"]
    test = [item for item in episodes if item.get("split") == "test"]
    if len(validation) != 10 or len(test) != 20:
        raise TeacherEvalBlocked("initial state split must be frozen 10/20")
    ids = [str(item["id"]) for item in episodes]
    if len(set(ids)) != len(ids):
        raise TeacherEvalBlocked("initial state ids are not unique")
    return {
        "schema": "owac.raw-teacher.initial-state-catalog.v1",
        "manifest": str(source),
        "states_path": str(states_path),
        "states_sha256": payload["states_sha256"],
        "episodes": episodes,
        "validation": validation,
        "test": test,
        "source_revision": payload.get("source_revision"),
        "target_xy_separation": {
            "from_source_starts": payload.get("target_xy_separation_from_50_source_starts_min"),
            "from_historical_inits": payload.get(
                "target_xy_separation_from_50_historical_inits_min"
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class Attempt:
    role: str
    episode_id: str
    split: str
    seed: int
    state_sha256: str
    status: str
    error: str | None = None


class AttemptLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._attempted: set[tuple[str, str]] = set()
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._attempted.add((str(record["role"]), str(record["episode_id"])))

    def ensure_available(self, attempt: Attempt) -> None:
        key = (attempt.role, attempt.episode_id)
        if key in self._attempted:
            raise TeacherEvalBlocked(f"duplicate/retried episode attempt: {key}")

    def begin(self, attempt: Attempt) -> None:
        key = (attempt.role, attempt.episode_id)
        self.ensure_available(attempt)
        self._attempted.add(key)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(attempt), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def complete(
        self,
        attempt: Attempt,
        *,
        success: bool | None,
        real_steps: int,
        replans: int,
        error: str | None = None,
        duration_seconds: float | None = None,
        status: str | None = None,
        valid_control_evidence: bool | None = None,
        interrupted: bool = False,
        inflight_uncertain: bool = False,
    ) -> None:
        if status is None:
            status = "completed" if success is True else "failed"
        record = {
            **asdict(attempt),
            "status": str(status),
            "success": success,
            "valid_control_evidence": (
                True if valid_control_evidence is None else bool(valid_control_evidence)
            ),
            "interrupted": bool(interrupted),
            "inflight_uncertain": bool(inflight_uncertain),
            "real_steps": int(real_steps),
            "replans": int(replans),
            "executed_actions_available": int(real_steps),
            "error": error,
            "duration_seconds": duration_seconds,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def paired_noise(*, seed: int, episode_id: str, frame: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{episode_id}:{frame}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.standard_normal((1, 10, 32), dtype=np.float32)


def fixed_denominators(
    *,
    validation_attempted: int,
    validation_success: int,
    test_attempted: int,
    test_success: int,
    reference_test_success: int | None = None,
    validation_fixed: int = 10,
    test_fixed: int = 20,
) -> dict[str, Any]:
    def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
        if total <= 0:
            return None, None
        z = 1.96
        p = successes / total
        denom = 1 + z * z / total
        centre = (p + z * z / (2 * total)) / denom
        half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    return {
        "validation": {
            "success": validation_success,
            "attempted": validation_attempted,
            "fixed_denominator": int(validation_fixed),
            "wilson_fixed": wilson(validation_success, int(validation_fixed)),
            "wilson_attempted": wilson(validation_success, validation_attempted),
        },
        "test": {
            "success": test_success,
            "attempted": test_attempted,
            "fixed_denominator": int(test_fixed),
            "wilson_fixed": wilson(test_success, int(test_fixed)),
            "wilson_attempted": wilson(test_success, test_attempted),
        },
        "total": {
            "success": validation_success + test_success,
            "attempted": validation_attempted + test_attempted,
        },
        "screen": {
            "teacher_at_least_16_of_20": test_success >= 16,
            "teacher_within_2_of_reference": (
                None
                if reference_test_success is None
                else test_success >= int(reference_test_success) - 2
            ),
            "not_statistical_noninferiority": True,
        },
    }


def read_native_loop_constants(
    path: str | Path = (
        "outputs/raw-distillation-20261020-sol/starvla-pinned-source/examples/"
        "simBenchmarks/LIBERO/eval_files/openpi/pi_libero_common.py"
    ),
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise TeacherEvalBlocked(f"pinned LIBERO common source absent: {source}")
    text = source.read_text()
    if "LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]" not in text:
        raise TeacherEvalBlocked("pinned dummy action contract changed")
    import importlib.util

    root = source.parents[5]
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("owac_pinned_libero_common_eval", source)
    if spec is None or spec.loader is None:
        raise TeacherEvalBlocked("cannot load pinned LIBERO common source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {
        "dummy_action": list(module.LIBERO_DUMMY_ACTION),
        "max_steps": {suite: int(module.get_max_steps(suite)) for suite in ("libero_goal",)},
        "source": str(source),
    }


__all__ = [
    "Attempt",
    "AttemptLedger",
    "MAX_TEST",
    "MAX_VALIDATION",
    "ROLES",
    "SETUP_WAIT_STEPS",
    "TeacherEvalBlocked",
    "fixed_denominators",
    "load_initial_states",
    "paired_noise",
    "read_native_loop_constants",
]


@dataclass(slots=True)
class EpisodeResult:
    episode_id: str
    role: str
    split: str
    success: bool | None
    steps: int
    replans: int
    error: str | None = None
    replan_metadata: list[dict[str, Any]] = field(default_factory=list)
    video_path: str | None = None
    result_files: list[str] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)
    status: str = "completed"
    valid_control_evidence: bool = True
    interrupted: bool = False
    inflight_uncertain: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def persist_episode_result(result: EpisodeResult, result_dir: str | Path) -> list[str]:
    """Write immutable episode JSON/NPZ; never overwrite an existing episode."""
    output = Path(result_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{result.role}_{result.episode_id}"
    json_path = output / f"{stem}.json"
    npz_path = output / f"{stem}.npz"
    if json_path.exists() or npz_path.exists():
        raise TeacherEvalBlocked(f"episode output already exists: {stem}")
    arrays: dict[str, np.ndarray] = {}
    if result.executed_actions:
        arrays["executed_actions"] = np.asarray(result.executed_actions, dtype=np.float32)
    clean_metadata = []
    for index, metadata in enumerate(result.replan_metadata):
        item = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                arrays[f"replan_{index}_{key}"] = value
                item[key] = {"array": f"replan_{index}_{key}", "shape": list(value.shape)}
            else:
                item[key] = _jsonable(value)
        clean_metadata.append(item)
    np.savez(npz_path, **arrays)
    full_times = [
        float(item["full_seconds"])
        for item in result.replan_metadata
        if isinstance(item.get("full_seconds"), (int, float))
    ]
    frontend_times = [
        float(item["frontend_seconds"])
        for item in result.replan_metadata
        if isinstance(item.get("frontend_seconds"), (int, float))
    ]
    timing = {
        "samples": len(full_times),
        "warmup_samples_excluded": 0,
        "full_act_p50_ms": float(np.percentile(full_times, 50) * 1000) if full_times else None,
        "full_act_p95_ms": float(np.percentile(full_times, 95) * 1000) if full_times else None,
        "frontend_p50_ms": float(np.percentile(frontend_times, 50) * 1000)
        if frontend_times
        else None,
        "frontend_p95_ms": float(np.percentile(frontend_times, 95) * 1000)
        if frontend_times
        else None,
        "peak_allocated_bytes": max(
            (
                int(item["peak_allocated_bytes"])
                for item in result.replan_metadata
                if item.get("peak_allocated_bytes") is not None
            ),
            default=None,
        ),
    }
    payload = {
        "schema": "owac.rawvla-teacher.episode-result.v1",
        "role": result.role,
        "episode_id": result.episode_id,
        "split": result.split,
        "success": result.success,
        "status": result.status,
        "valid_control_evidence": result.valid_control_evidence,
        "interrupted": result.interrupted,
        "inflight_uncertain": result.inflight_uncertain,
        "steps": result.steps,
        "replans": result.replans,
        "executed_actions_available": len(result.executed_actions),
        "error": result.error,
        "video_path": result.video_path,
        "replan_metadata": clean_metadata,
        "arrays_npz": str(npz_path),
        "timing": timing,
        "action_protocol": "sampled10/execute5",
    }
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return [str(json_path), str(npz_path)]


class ClosedLoopDriver:
    """Role-neutral closed-loop execution contract.

    The caller supplies a LIBERO environment factory and a controller with
    ``reset()``/``act(ring, robot_state, noise)``. No environment is imported or
    instantiated here. Both roles use the same initial state, seed/noise,
    lighting realization and chunk10/execute5 schedule.
    """

    def __init__(
        self,
        *,
        catalog: dict[str, Any],
        budget: Any,
        ledger: AttemptLedger,
        max_steps: int = 300,
        setup_wait_steps: int = SETUP_WAIT_STEPS,
        chunk_size: int = 10,
        execute_steps: int = 5,
        media_dir: str | Path | None = None,
        result_dir: str | Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.budget = budget
        self.ledger = ledger
        self.max_steps = int(max_steps)
        self.setup_wait_steps = int(setup_wait_steps)
        self.chunk_size = int(chunk_size)
        self.execute_steps = int(execute_steps)
        self.media_dir = Path(media_dir) if media_dir is not None else None
        self.result_dir = Path(result_dir) if result_dir is not None else None
        if self.chunk_size != 10 or self.execute_steps != 5:
            raise TeacherEvalBlocked("closed-loop contract requires chunk10/execute5")

    def _states(self, split: str) -> list[dict[str, Any]]:
        if split not in {"validation", "test"}:
            raise TeacherEvalBlocked("closed-loop split must be validation or test")
        states = self.catalog[split]
        limit = MAX_VALIDATION if split == "validation" else MAX_TEST
        if len(states) > limit:
            raise TeacherEvalBlocked("initial-state split exceeds hard cap")
        return states

    def run_role(
        self,
        *,
        role: str,
        split: str,
        env_factory: Any,
        controller: Any,
        states_array: Any,
        seed: int = 101,
    ) -> list[EpisodeResult]:
        if role not in ROLES:
            raise TeacherEvalBlocked(f"unknown role: {role}")
        if role == "default_reference" and split == "validation":
            raise TeacherEvalBlocked("default reference validation episodes are not allowed")
        states = self._states(split)
        if not states:
            return []
        probe = env_factory(states[0], None, role)
        if hasattr(probe, "validate"):
            probe.validate()
        results: list[EpisodeResult] = []
        for state_entry in states:
            episode_id = str(state_entry["id"])
            started = time.monotonic()
            if self.media_dir is not None:
                candidate = self.media_dir / f"{role}_{episode_id}.mp4"
                if candidate.exists():
                    raise TeacherEvalBlocked(f"video already exists: {candidate}")
            if self.result_dir is not None:
                for suffix in (".json", ".npz"):
                    candidate = self.result_dir / f"{role}_{episode_id}{suffix}"
                    if candidate.exists():
                        raise TeacherEvalBlocked(f"episode output already exists: {candidate}")
            state_index = int(state_entry["array_index"])
            state_hash = hashlib.sha256(
                np.asarray(states_array[state_index], dtype=np.float64).tobytes()
            ).hexdigest()
            if state_hash != str(state_entry["state_sha256"]):
                raise TeacherEvalBlocked(f"initial-state hash differs for {episode_id}")
            attempt = Attempt(
                role=role,
                episode_id=episode_id,
                split=split,
                seed=int(seed),
                state_sha256=str(state_entry["state_sha256"]),
                status="running",
            )
            self.ledger.ensure_available(attempt)
            self.budget.check_deadline()
            if hasattr(self.budget, "check_disk"):
                self.budget.check_disk(reserve_bytes=10 * 1024 * 1024)
            self.budget.charge_policy_episode(split)
            self.ledger.begin(attempt)
            env = None
            replans = 0
            steps = 0
            video_path = None
            media = None
            try:
                controller.reset()
                if self.media_dir is not None:
                    self.media_dir.mkdir(parents=True, exist_ok=True)
                    video_path = str(self.media_dir / f"{role}_{episode_id}.mp4")
                    import imageio.v2 as imageio

                    media = imageio.get_writer(video_path, fps=10, codec="libx264")
                env = env_factory(state_entry, states_array, role)
                observation = env.reset()
                if hasattr(env, "lighting") and hasattr(controller, "lighting"):
                    controller.lighting = env.lighting or {}
                for wait_index in range(self.setup_wait_steps):
                    observation, _reward, _done, _info = env.step(list([0.0] * 6 + [-1.0]))
                    if hasattr(controller, "observe"):
                        controller.observe(observation, wait_index + 1)
                ring: list[Any] = []
                plan: Any = None
                replan_metadata: list[dict[str, Any]] = []
                executed_actions: list[np.ndarray] = []
                while steps < self.max_steps:
                    self.budget.check_deadline()
                    raw = observation.get("raw")
                    if raw is None:
                        raise TeacherEvalBlocked("environment did not provide RAW frame")
                    physical_frame = self.setup_wait_steps + steps
                    if hasattr(controller, "observe"):
                        controller.observe(observation, physical_frame)
                    if media is not None:
                        native = observation.get("native") or {}
                        frame = native.get("agentview_image")
                        if frame is not None:
                            media.append_data(np.asarray(frame, dtype=np.uint8))
                    ring.append(raw)
                    ring = ring[-6:]
                    if steps % self.execute_steps == 0:
                        noise = paired_noise(seed=seed, episode_id=episode_id, frame=steps)
                        plan, metadata = controller.act(
                            observation,
                            observation.get("robot_state"),
                            noise,
                            episode_id=episode_id,
                            policy_step=self.setup_wait_steps + steps,
                        )
                        replan_metadata.append(metadata)
                        replans += 1
                    action = plan[steps % self.execute_steps] if plan is not None else [0.0] * 7
                    observation, _reward, done, _info = env.step(list(action))
                    executed_actions.append(np.asarray(action, dtype=np.float32).copy())
                    steps += 1
                    if done or bool(env.check_success()):
                        break
                success = bool(env.check_success())
                if media is not None:
                    media.close()
                    media = None
                self.ledger.complete(
                    attempt,
                    success=success,
                    real_steps=steps,
                    replans=replans,
                    duration_seconds=time.monotonic() - started,
                )
                episode_result = EpisodeResult(
                    episode_id,
                    role,
                    split,
                    success,
                    steps,
                    replans,
                    None,
                    replan_metadata,
                    video_path,
                    executed_actions=executed_actions,
                    status="completed",
                    valid_control_evidence=True,
                )
                if self.result_dir is not None:
                    episode_result.result_files = persist_episode_result(
                        episode_result, self.result_dir
                    )
                print(
                    f"[{role}] {episode_id}: success={episode_result.success} "
                    f"steps={episode_result.steps} replans={episode_result.replans}",
                    flush=True,
                )
                results.append(episode_result)
            except (KeyboardInterrupt, SystemExit) as exc:
                # Persist the partial, known evidence before preserving the
                # process-level stop.  The in-flight action is not guessed.
                if media is not None:
                    try:
                        media.close()
                    except Exception:
                        pass
                    media = None
                message = f"{type(exc).__name__}: {exc}"
                try:
                    self.ledger.complete(
                        attempt,
                        success=None,
                        real_steps=steps,
                        replans=replans,
                        error=message,
                        duration_seconds=time.monotonic() - started,
                        status="interrupted",
                        valid_control_evidence=False,
                        interrupted=True,
                        inflight_uncertain=True,
                    )
                    episode_result = EpisodeResult(
                        episode_id,
                        role,
                        split,
                        None,
                        steps,
                        replans,
                        message,
                        replan_metadata,
                        video_path,
                        executed_actions=executed_actions,
                        status="interrupted",
                        valid_control_evidence=False,
                        interrupted=True,
                        inflight_uncertain=True,
                    )
                    if self.result_dir is not None:
                        episode_result.result_files = persist_episode_result(
                            episode_result, self.result_dir
                        )
                finally:
                    raise exc from None
            except Exception as exc:
                if media is not None:
                    try:
                        media.close()
                    except Exception:
                        pass
                    media = None
                message = f"{type(exc).__name__}: {exc}"
                self.ledger.complete(
                    attempt,
                    success=False,
                    real_steps=steps,
                    replans=replans,
                    error=message,
                    duration_seconds=time.monotonic() - started,
                    status="failed",
                    valid_control_evidence=False,
                )
                episode_result = EpisodeResult(
                    episode_id,
                    role,
                    split,
                    False,
                    steps,
                    replans,
                    message,
                    replan_metadata,
                    video_path,
                    executed_actions=executed_actions,
                    status="failed",
                    valid_control_evidence=False,
                )
                if self.result_dir is not None:
                    episode_result.result_files = persist_episode_result(
                        episode_result, self.result_dir
                    )
                print(
                    f"[{role}] {episode_id}: success={episode_result.success} "
                    f"steps={episode_result.steps} replans={episode_result.replans}",
                    flush=True,
                )
                results.append(episode_result)
                if steps == 0:
                    break
            finally:
                if media is not None:
                    try:
                        media.close()
                    except Exception:
                        pass
                if env is not None and hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass
        return results


def run_paired_closed_loop(
    *,
    teacher: Any,
    stats: Any,
    catalog: dict[str, Any],
    states_array: Any,
    budget: Any,
    ledger_path: str | Path,
    author_root: str | Path,
    libero_config: str | Path,
    split: str = "test",
    seed: int = 101,
    identity: dict[str, Any] | None = None,
    result_dir: str | Path | None = None,
    media_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Concrete callable binding both roles to the same frozen states/noise.

    The caller must already own a PhaseBudget GPU lease covering model load,
    both roles and media writing. This function performs no model loading.
    """
    from owac.simulation.rawvla_teacher_controller import (
        LIBEROEnvAdapter,
        TeacherPolicyController,
    )

    if identity is None:
        raise TeacherEvalBlocked("paired closed loop requires caller model identity")
    constants = read_native_loop_constants()
    if constants["max_steps"]["libero_goal"] != 300:
        raise TeacherEvalBlocked("native LIBERO goal max steps changed")
    if split == "validation":
        roles = ("learned_raw_teacher",)
    elif split == "test":
        roles = ROLES
    else:
        raise TeacherEvalBlocked("paired closed-loop split must be validation or test")
    ledger = AttemptLedger(ledger_path)
    result_root = (
        Path(result_dir) if result_dir is not None else Path(ledger_path).parent / "paired-results"
    )
    result_root.mkdir(parents=True, exist_ok=False)
    media_root = Path(media_dir) if media_dir is not None else result_root / "media"
    media_root.mkdir(parents=True, exist_ok=False)
    driver = ClosedLoopDriver(
        catalog=catalog,
        budget=budget,
        ledger=ledger,
        media_dir=media_root,
        result_dir=result_root / "episodes",
    )
    results: dict[str, list[EpisodeResult]] = {}
    for role in roles:
        controller = TeacherPolicyController(
            teacher=teacher,
            stats=stats,
            role=role,
            author_root=author_root,
            seed=seed,
        )
        adapter = LIBEROEnvAdapter(
            author_root=author_root,
            libero_config=libero_config,
        )

        def env_factory(state_entry, states, role_name, *, _adapter=adapter):
            return _adapter.bind(state_entry, states)

        results[role] = driver.run_role(
            role=role,
            split=split,
            env_factory=env_factory,
            controller=controller,
            states_array=states_array,
            seed=seed,
        )
    summary = summarize_paired_results(results, catalog)
    summary["identity"] = dict(identity or {})
    summary["media_root"] = str(media_root)
    summary["result_root"] = str(result_root)
    summary_path = result_root / "paired-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "owac.raw-teacher.paired-closed-loop.v1",
        "summary": summary,
        "summary_json": str(summary_path),
        "episode_files": {
            role: [path for result in episodes for path in result.result_files]
            for role, episodes in results.items()
        },
    }


__all__ += [
    "ClosedLoopDriver",
    "EpisodeResult",
    "run_paired_closed_loop",
    "summarize_paired_results",
]


def summarize_paired_results(
    results: dict[str, list[EpisodeResult]], catalog: dict[str, Any]
) -> dict[str, Any]:
    """Fixed-denominator paired report; unattempted states remain explicit."""
    intended = {
        "validation": [str(item["id"]) for item in catalog.get("validation", [])],
        "test": [str(item["id"]) for item in catalog.get("test", [])],
    }
    summary: dict[str, Any] = {"schema": "owac.raw-teacher.paired-summary.v1"}
    for role in ROLES:
        episodes = results.get(role, [])
        by_id = {item.episode_id: item for item in episodes}
        summary[role] = {}
        for split in ("validation", "test"):
            ids = intended[split]
            attempted = [episode_id for episode_id in ids if episode_id in by_id]
            successes = sum(bool(by_id[episode_id].success) for episode_id in attempted)
            summary[role][split] = {
                "fixed_denominator": len(ids),
                "attempted": len(attempted),
                "success": successes,
                "unattempted": [episode_id for episode_id in ids if episode_id not in by_id],
                "errors": [
                    {"id": episode_id, "error": by_id[episode_id].error}
                    for episode_id in attempted
                    if by_id[episode_id].error
                ],
            }
    ref = summary.get("default_reference", {}).get("test", {})
    teacher = summary.get("learned_raw_teacher", {}).get("test", {})
    ref_by_id = {item.episode_id: item for item in results.get("default_reference", [])}
    teacher_by_id = {item.episode_id: item for item in results.get("learned_raw_teacher", [])}
    paired = [
        episode_id
        for episode_id in intended["test"]
        if episode_id in ref_by_id and episode_id in teacher_by_id
    ]
    summary["paired_test"] = {
        "ids": paired,
        "regressions": [
            episode_id
            for episode_id in paired
            if ref_by_id[episode_id].success and not teacher_by_id[episode_id].success
        ],
        "recoveries": [
            episode_id
            for episode_id in paired
            if teacher_by_id[episode_id].success and not ref_by_id[episode_id].success
        ],
    }
    denominators = fixed_denominators(
        validation_attempted=summary["learned_raw_teacher"]["validation"]["attempted"],
        validation_success=summary["learned_raw_teacher"]["validation"]["success"],
        test_attempted=teacher["attempted"],
        test_success=teacher["success"],
        reference_test_success=ref["success"],
        validation_fixed=len(intended["validation"]),
        test_fixed=len(intended["test"]),
    )
    summary["denominators"] = denominators
    summary["screen"] = denominators["screen"]
    return summary
