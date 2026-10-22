"""Authorized C-only continuation of the actual C50 RAW->RAW token distillation.

This module owns the narrow migration from the completed pilot recipe to the
authorized 5000-update continuation.  It never renames the old pilot recipe or
silently relaxes identity checks:

1. the actual parent checkpoint SHA and the parent config SHA are verified;
2. the parent checkpoint is loaded with the *parent* config and parent run
   identity, restoring optimizer, scheduler, sampler, RNG and counters;
3. only the explicit operational whitelist in the migration manifest may differ
   between the parent recipe digest and the new recipe digest;
4. the new run keeps the same student, numerical recipe, data identity and
   optimizer/scheduler objects, performs its next actual update at step 51, and
   binds the new recipe plus parent provenance into every new checkpoint.

Selection is held-out projected-token ``MSE + 0.1 * cosine`` over the fixed 102
validation windows / 5 validation sources, minimum total, earlier tie.  Action
MAE is an optional offline diagnostic and is structurally unable to select or
stop training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CHECKPOINT_SCHEMA",
    "MIGRATION_SCHEMA",
    "ContinuationBudget",
    "MigrationValidation",
    "owned_run_bytes",
    "plan_checkpoint_retention",
    "recipe_digest",
    "validate_continuation_record",
    "validate_migration_manifest",
    "validate_parent_checkpoint",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_SCHEMA = "owac.raw-stem-temporal.c-full.migration.v1"
CHECKPOINT_SCHEMA = "owac.raw-stem-temporal.c-full.continuation.v1"
RUN_STATUS_SCHEMA = "owac.raw-stem-temporal.c-full.run-status.v1"
RUN_META_SCHEMA = "owac.raw-stem-temporal.c-full.run-meta.v1"
HEARTBEAT_SCHEMA = "owac.raw-stem-temporal.c-full.heartbeat.v1"
UPDATE_SCHEMA = "owac.raw-stem-temporal.c-full.update.v1"
VALIDATION_SCHEMA = "owac.raw-stem-temporal.c-full.validation.v1"
SELECTION_SCHEMA = "owac.raw-stem-temporal.feature-selection.v1"
PARENT_BASELINE_SCHEMA = "owac.raw-stem-temporal.c-full.parent-baseline.v1"
PARENT_HISTORY_SCHEMA = "owac.raw-stem-temporal.c-full.parent-history.v1"
PREPARED_SCHEMA = "owac.raw-stem-temporal.c-full.prepared.v1"
ACTION_DIAGNOSTIC_SCHEMA = "owac.raw-stem-temporal.c-full.action-diagnostic.v1"

TOTAL_UPDATES = 5000
PARENT_UPDATES = 50
TOTAL_ANCHORS = 80000
PARENT_ANCHORS = 800
NEW_WALL_SECONDS = 230400
NEW_OUTPUT_BYTES = 21474836480
VALIDATION_INTERVAL = 250
MILESTONE_STEPS = (500, 2000, 5000)
CHECKPOINT_RESERVE_FRACTION = 1.5

# Explicitly allowed operational changes.  Everything else is numerical/model/
# data/history identity and must be byte-for-byte equal between old and new.
OPERATIONAL_WHITELIST = (
    "authorization",
    "assets.output_root",
    "continuation",
    "training.pilot_max_updates_per_arm",
    "training.checkpoint_updates",
    "training.checkpoint_interval_updates",
    "training.earlystop_patience",
    "training.max_wall_seconds",
    "training.max_output_bytes",
    "training.combined_gpu_hours",
    "training.recovery_checkpoints",
    "training.selection_metric",
    "evaluation.selection_metric",
)

NUMERICAL_IDENTITY_PATHS = (
    "training.seed",
    "training.effective_batch",
    "training.microbatch",
    "training.max_cumulative_updates_per_arm",
    "training.max_anchor_presentations_per_arm",
    "training.learning_rate",
    "training.optimizer",
    "training.weight_decay",
    "training.gradient_clip",
    "training.warmup_updates",
    "training.activation_memory_mode",
    "training.lambda_f",
    "training.cosine_weight",
    "training.lambda_a_D",
    "training.stage1",
    "training.stage2",
    "student",
    "task",
    "teacher_requirement",
    "evaluation.flow_steps",
    "evaluation.action_horizon",
    "evaluation.execute_steps",
    "evaluation.selection_split",
    "evaluation.test_selection",
    "evaluation.groups",
    "evaluation.controls",
    "evaluation.target_position_intervention",
)


# --------------------------------------------------------------------------- #
# Small durable-file helpers
# --------------------------------------------------------------------------- #
class _ResourceStop(RuntimeError):
    """A cap was hit at a checkpoint boundary; the final state must still persist."""


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    _fsync_dir(target.parent)
    return target


def _atomic_text(path: str | Path, value: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    _fsync_dir(target.parent)
    return target


def _append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, allow_nan=False)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl_tolerant(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            # A killed append may leave a torn final line; earlier lines stand.
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _proc_start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def _git_head() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


# --------------------------------------------------------------------------- #
# Recipe identity and narrow-diff validation
# --------------------------------------------------------------------------- #
def recipe_digest(config: Mapping[str, Any]) -> str:
    """Canonical recipe digest; authorization and output directory excluded."""
    value = copy.deepcopy(dict(config))
    value.pop("authorization", None)
    assets = value.get("assets")
    if isinstance(assets, Mapping):
        assets.pop("output_root", None)
    return _sha256_text(_canonical_json(value))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        if not value:
            result[prefix] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
    else:
        result[prefix] = value
    return result


def config_diff(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    old_flat = _flatten(old)
    new_flat = _flatten(new)
    paths = set(old_flat) | set(new_flat)
    changed = []
    for path in sorted(paths):
        if path not in old_flat or path not in new_flat:
            changed.append(path)
        elif old_flat[path] != new_flat[path]:
            changed.append(path)
    return changed


def _path_allowed(path: str, allowed: Sequence[str]) -> bool:
    for item in allowed:
        if path == item or path.startswith(item + ".") or item.startswith(path + "."):
            return True
    return False


def _required_identity(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    old_flat = _flatten(old)
    new_flat = _flatten(new)
    for path in NUMERICAL_IDENTITY_PATHS:
        if old_flat.get(path) != new_flat.get(path):
            raise ValueError(f"migration would change numerical/model/data identity: {path}")
    old_assets = dict(old.get("assets") or {})
    new_assets = dict(new.get("assets") or {})
    old_assets.pop("output_root", None)
    new_assets.pop("output_root", None)
    if old_assets != new_assets:
        raise ValueError("migration would change an asset identity/path other than output_root")
    if old.get("schema") != new.get("schema"):
        raise ValueError("migration would change the config schema")
    if old.get("candidate") != new.get("candidate") or old.get("decision") != new.get("decision"):
        raise ValueError("migration would change candidate/decision identity")


def _parse_iso(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an ISO8601 string")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not ISO8601") from exc
    return value


def _require_sha(value: Any, name: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{name} must be a 64-character lowercase SHA256")
    return text


@dataclass(frozen=True, slots=True)
class ContinuationBudget:
    total_updates: int = TOTAL_UPDATES
    parent_updates: int = PARENT_UPDATES
    total_anchors: int = TOTAL_ANCHORS
    parent_anchors: int = PARENT_ANCHORS
    max_new_wall_seconds: int = NEW_WALL_SECONDS
    max_new_output_bytes: int = NEW_OUTPUT_BYTES

    @property
    def max_new_updates(self) -> int:
        return self.total_updates - self.parent_updates

    @property
    def max_new_anchors(self) -> int:
        return self.total_anchors - self.parent_anchors


def budget_from_user_authorization(path: str | Path) -> ContinuationBudget:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("user authorization must be an object")
    if payload.get("schema") != "owac.raw-distillation.c-full-authorization.v1":
        raise ValueError("user authorization schema differs")
    if payload.get("closed_loop") is not False:
        raise ValueError("this continuation does not authorize closed-loop episodes")
    total_updates = int(payload["max_cumulative_updates"])
    parent_updates = int(payload["already_completed_updates"])
    total_anchors = int(payload["max_cumulative_anchor_presentations"])
    new_anchors = int(payload["max_new_anchor_presentations"])
    parent_anchors = int(
        payload.get(
            "already_completed_anchor_presentations",
            total_anchors - new_anchors,
        )
    )
    budget = ContinuationBudget(
        total_updates=total_updates,
        parent_updates=parent_updates,
        total_anchors=total_anchors,
        parent_anchors=parent_anchors,
        max_new_wall_seconds=int(payload["max_new_gpu_wall_seconds"]),
        max_new_output_bytes=int(payload["max_new_output_bytes"]),
    )
    if int(payload["max_new_updates"]) != budget.max_new_updates:
        raise ValueError("user authorization new-update cap does not equal total-parent")
    if int(payload["max_new_anchor_presentations"]) != budget.max_new_anchors:
        raise ValueError("user authorization new-anchor cap does not equal total-parent")
    if budget != ContinuationBudget():
        raise ValueError("user authorization caps differ from the frozen C-full contract")
    return budget


@dataclass(frozen=True, slots=True)
class MigrationValidation:
    manifest_path: Path
    manifest_sha256: str
    parent_checkpoint: Path
    parent_checkpoint_sha256: str
    parent_config: Path
    parent_config_sha256: str
    new_config: Path
    new_config_sha256: str
    old_recipe_sha256: str
    new_recipe_sha256: str
    allowed_operational_changes: tuple[str, ...]
    carried_receipts: tuple[Mapping[str, Any], ...]
    user_authorization: Path
    run_root: Path
    parent_assessment: Path
    parent_assessment_sha256: str
    baseline_tolerance_abs: float
    expected_parent_baseline: Mapping[str, Any]


def _resolve(path_value: Any, *, base: Path) -> Path:
    candidate = Path(str(path_value))
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def validate_migration_manifest(
    manifest_path: str | Path,
    *,
    new_config_path: str | Path,
    parent_config_path: str | Path,
    parent_checkpoint_path: str | Path,
    user_authorization_path: str | Path,
    run_root: str | Path | None = None,
) -> MigrationValidation:
    """Validate the explicit narrow migration, without touching the checkpoint."""
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_json(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError("migration manifest must be an object")
    if manifest.get("schema") != MIGRATION_SCHEMA:
        raise ValueError("migration manifest schema differs")
    if manifest.get("authorized") is not True:
        raise ValueError("migration manifest is not authorized")
    if manifest.get("arm") != "C" or manifest.get("c_only") is not True:
        raise ValueError("migration is C-only")
    for key in ("b_arm", "d_arm", "vision_unfreeze", "action_loss", "closed_loop", "download"):
        if manifest.get(key) is not False:
            raise ValueError(f"migration must explicitly refuse {key}")
    if not str(manifest.get("authorized_by") or "").strip():
        raise ValueError("migration manifest needs authorized_by")
    _parse_iso(manifest.get("authorized_at"), "authorized_at")

    parent_config = Path(parent_config_path).resolve()
    new_config = Path(new_config_path).resolve()
    parent_checkpoint = Path(parent_checkpoint_path).resolve()
    user_authorization = Path(user_authorization_path).resolve()
    if not parent_config.is_file() or not new_config.is_file() or not parent_checkpoint.is_file():
        raise ValueError("migration parent/new config or parent checkpoint is absent")
    if not user_authorization.is_file():
        raise ValueError("user authorization is absent")
    if manifest.get("parent_config") is not None:
        declared = _resolve(manifest["parent_config"], base=REPO_ROOT)
        if declared != parent_config:
            raise ValueError("migration parent_config path differs")
    if manifest.get("new_config") is not None:
        declared = _resolve(manifest["new_config"], base=REPO_ROOT)
        if declared != new_config:
            raise ValueError("migration new_config path differs")
    if manifest.get("parent_checkpoint") is not None:
        declared = _resolve(manifest["parent_checkpoint"], base=REPO_ROOT)
        if declared != parent_checkpoint:
            raise ValueError("migration parent_checkpoint path differs")

    parent_config_sha = _sha256_file(parent_config)
    new_config_sha = _sha256_file(new_config)
    parent_checkpoint_sha = _sha256_file(parent_checkpoint)
    for name, actual in (
        ("parent_config_sha256", parent_config_sha),
        ("new_config_sha256", new_config_sha),
        ("parent_checkpoint_sha256", parent_checkpoint_sha),
    ):
        expected = _require_sha(manifest.get(name), name)
        if expected != actual:
            raise ValueError(f"migration {name} differs from the actual file")

    old_config = _load_json(parent_config)
    new_config_value = _load_json(new_config)
    if not isinstance(old_config, Mapping) or not isinstance(new_config_value, Mapping):
        raise ValueError("migration configs must be objects")
    old_recipe = recipe_digest(old_config)
    new_recipe = recipe_digest(new_config_value)
    if manifest.get("old_recipe_sha256") != old_recipe:
        raise ValueError("migration old_recipe_sha256 differs from the parent config digest")
    if manifest.get("new_recipe_sha256") != new_recipe:
        raise ValueError("migration new_recipe_sha256 differs from the new config digest")
    if old_recipe == new_recipe:
        raise ValueError("migration must change the recipe digest for the new authority")

    allowed_raw = manifest.get("allowed_operational_changes")
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise ValueError("migration must explicitly list allowed operational changes")
    allowed = tuple(str(item) for item in allowed_raw)
    for item in allowed:
        if item not in OPERATIONAL_WHITELIST:
            raise ValueError(f"migration lists a non-whitelisted change: {item}")
    diff = config_diff(old_config, new_config_value)
    if not diff:
        raise ValueError("new config does not differ from the parent config")
    for path in diff:
        if not _path_allowed(path, allowed):
            raise ValueError(f"migration changes a non-authorized path: {path}")
        if not _path_allowed(path, OPERATIONAL_WHITELIST):
            raise ValueError(f"migration path is outside the operational whitelist: {path}")
    _required_identity(old_config, new_config_value)

    budget = budget_from_user_authorization(user_authorization)
    training = new_config_value.get("training") or {}
    if int(training.get("max_cumulative_updates_per_arm", -1)) != budget.total_updates:
        raise ValueError("new config cumulative update cap differs from the authority")
    if int(training.get("max_anchor_presentations_per_arm", -1)) != budget.total_anchors:
        raise ValueError("new config anchor cap differs from the authority")
    if int(training.get("max_wall_seconds", -1)) != budget.max_new_wall_seconds:
        raise ValueError("new config new-stage wall cap differs from the authority")
    if int(training.get("max_output_bytes", -1)) != budget.max_new_output_bytes:
        raise ValueError("new config new-output cap differs from the authority")
    if int(training.get("checkpoint_interval_updates", -1)) not in (100, 250):
        raise ValueError("checkpoint interval must be 100 or 250")
    if int(training.get("recovery_checkpoints", -1)) != 2:
        raise ValueError("checkpoint recovery retention must be exactly 2")
    if training.get("selection_metric") != "validation_token_total":
        raise ValueError("training selection_metric must be validation_token_total")
    evaluation = new_config_value.get("evaluation") or {}
    if evaluation.get("selection_metric") != "validation_token_total":
        raise ValueError("evaluation selection_metric must be validation_token_total")
    if evaluation.get("test_selection") is not False:
        raise ValueError("test selection must remain false")
    authorization = new_config_value.get("authorization") or {}
    if authorization.get("optimizer_updates") is not True:
        raise ValueError("new config must authorize optimizer updates")
    if authorization.get("closed_loop") not in (False, None):
        raise ValueError("new config must refuse closed loop")

    run_root_path = Path(run_root or new_config_value["assets"]["output_root"]).resolve()
    declared_run_root = _resolve(
        (new_config_value.get("assets") or {}).get("output_root"), base=REPO_ROOT
    )
    if run_root_path != declared_run_root:
        raise ValueError("run root differs from the configured output root")
    expected_assets_output = declared_run_root
    if (
        manifest.get("run_root") is not None
        and _resolve(manifest["run_root"], base=REPO_ROOT) != run_root_path
    ):
        raise ValueError("migration run_root differs from the configured output root")
    if expected_assets_output.parent != (REPO_ROOT / "outputs").resolve():
        raise ValueError("run root must be a direct child of the repository outputs directory")
    parent_run_root = parent_checkpoint.parents[1].resolve()
    if run_root_path == parent_run_root or parent_run_root in run_root_path.parents:
        raise ValueError("continuation must not write inside the parent pilot run root")

    assessment_path = _resolve(
        manifest.get("parent_assessment")
        or (
            "outputs/raw-distillation-pilot-20261021-101725/"
            "postprocess/run-20261021T035315Z/attached-assessment.json"
        ),
        base=REPO_ROOT,
    )
    if not assessment_path.is_file():
        raise ValueError("parent C50 feature assessment is absent")
    assessment_sha = _sha256_file(assessment_path)
    if manifest.get("parent_assessment_sha256") is not None:
        if (
            _require_sha(manifest["parent_assessment_sha256"], "parent_assessment_sha256")
            != assessment_sha
        ):
            raise ValueError("migration parent assessment SHA differs")
    assessment = _load_json(assessment_path)
    try:
        expected_global = assessment["arms"]["C"]["steps"]["50"]["feature_targets"]["global"]
    except (KeyError, TypeError) as exc:
        raise ValueError("parent assessment lacks actual C50 feature globals") from exc
    for key in ("feature_mse", "feature_cosine", "feature_total", "scalar_denominator"):
        if key not in expected_global:
            raise ValueError(f"parent assessment lacks {key}")
    if (
        int(expected_global.get("anchors", -1)) != 102
        or int(expected_global.get("valid_tokens", -1)) != 52224
    ):
        raise ValueError("parent assessment C50 token/window denominators differ")
    if float(expected_global.get("scalar_denominator", -1)) != 106954752.0:
        raise ValueError("parent assessment C50 scalar denominator differs")
    tolerance = float(manifest.get("parent_assessment_tolerance_abs", 1e-6))
    if tolerance != 1e-6:
        raise ValueError("parent baseline tolerance is frozen at 1e-6; do not loosen it")
    if int(training.get("earlystop_patience", -1)) != 0:
        raise ValueError("action early stopping must be disabled (earlystop_patience=0)")

    declared_caps = manifest.get("declared_caps")
    expected_caps = {
        "max_cumulative_updates": budget.total_updates,
        "max_new_updates": budget.max_new_updates,
        "max_cumulative_anchor_presentations": budget.total_anchors,
        "max_new_anchor_presentations": budget.max_new_anchors,
        "max_new_gpu_wall_seconds": budget.max_new_wall_seconds,
        "max_new_output_bytes": budget.max_new_output_bytes,
    }
    if declared_caps != expected_caps:
        raise ValueError("migration declared_caps differ from the actual user authorization")

    carried_raw = manifest.get("carried_readiness_receipts")
    if not isinstance(carried_raw, list):
        raise ValueError("migration must explicitly list carried readiness receipts")
    carried: list[Mapping[str, Any]] = []
    seen_kinds: set[str] = set()
    for entry in carried_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("carried receipt entry must be an object")
        kind = str(entry.get("kind") or "")
        if kind not in {"parity", "resource"} or kind in seen_kinds:
            raise ValueError("carried receipts must contain one parity and one resource entry")
        seen_kinds.add(kind)
        receipt_path = Path(str(entry.get("path") or ""))
        if not receipt_path.is_absolute():
            receipt_path = (REPO_ROOT / receipt_path).resolve()
        else:
            receipt_path = receipt_path.resolve()
        if not receipt_path.is_file():
            raise ValueError(f"carried {kind} readiness receipt is absent")
        expected_sha = _require_sha(entry.get("sha256"), f"carried {kind} sha256")
        if _sha256_file(receipt_path) != expected_sha:
            raise ValueError(f"carried {kind} readiness receipt SHA differs")
        receipt = _load_json(receipt_path)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"carried {kind} receipt is not an object")
        if receipt.get("passed") is not True or int(receipt.get("optimizer_updates", -1)) != 0:
            raise ValueError(f"carried {kind} receipt is not a real zero-update pass")
        if receipt.get("research_config_sha256") not in {old_recipe, new_recipe}:
            raise ValueError(f"carried {kind} receipt recipe identity differs")
        carried.append(
            {
                "kind": kind,
                "path": str(receipt_path),
                "sha256": expected_sha,
                "protocol": receipt.get("protocol"),
                "research_config_sha256": receipt.get("research_config_sha256"),
                "passed": True,
                "rewritten": False,
            }
        )
    if seen_kinds != {"parity", "resource"}:
        raise ValueError("both parity and resource readiness receipts are required")

    manifest_sha = _sha256_file(manifest_file)
    return MigrationValidation(
        manifest_path=manifest_file,
        manifest_sha256=manifest_sha,
        parent_checkpoint=parent_checkpoint,
        parent_checkpoint_sha256=parent_checkpoint_sha,
        parent_config=parent_config,
        parent_config_sha256=parent_config_sha,
        new_config=new_config,
        new_config_sha256=new_config_sha,
        old_recipe_sha256=old_recipe,
        new_recipe_sha256=new_recipe,
        allowed_operational_changes=allowed,
        carried_receipts=tuple(carried),
        user_authorization=user_authorization,
        run_root=run_root_path,
        parent_assessment=assessment_path,
        parent_assessment_sha256=assessment_sha,
        baseline_tolerance_abs=tolerance,
        expected_parent_baseline=dict(expected_global),
    )


def validate_parent_checkpoint(
    payload: Mapping[str, Any],
    *,
    parent_checkpoint_sha256: str,
    old_recipe_sha256: str,
    parent_config_sha256: str,
    expected_updates: int = PARENT_UPDATES,
    expected_anchors: int = PARENT_ANCHORS,
) -> dict[str, Any]:
    """Strict identity/state audit of the actual C50 payload before migration."""
    _require_sha(parent_checkpoint_sha256, "parent_checkpoint_sha256")
    _require_sha(old_recipe_sha256, "old_recipe_sha256")
    _require_sha(parent_config_sha256, "parent_config_sha256")
    if payload.get("schema") != "owac.raw-stem-temporal.checkpoint.v1":
        raise ValueError("parent checkpoint schema differs")
    if payload.get("arm") != "C":
        raise ValueError("parent checkpoint is not the C arm")
    if payload.get("scope") not in {"stem_gate", "stem_plus_gate"}:
        raise ValueError("parent checkpoint scope differs")
    updates = int(payload.get("cumulative_updates", -1))
    anchors = int(payload.get("anchor_presentations", -1))
    if updates != expected_updates or anchors != expected_anchors:
        raise ValueError("parent checkpoint is not the completed C50/800 state")
    identity = payload.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("parent checkpoint lacks run identity")
    if identity.get("research_config_sha256") != old_recipe_sha256:
        raise ValueError("parent checkpoint old recipe SHA differs")
    if identity.get("scope") != "stem_plus_gate":
        raise ValueError("parent checkpoint run-identity scope differs")
    if not isinstance(identity.get("cache_identity_key"), str):
        raise ValueError("parent checkpoint lacks cache identity")
    training = payload.get("training_config")
    if not isinstance(training, Mapping):
        raise ValueError("parent checkpoint lacks the full training recipe")
    if int(training.get("max_cumulative_updates_per_arm", -1)) != TOTAL_UPDATES:
        raise ValueError("parent checkpoint cumulative recipe cap differs")
    if int(training.get("pilot_max_updates_per_arm", -1)) != PARENT_UPDATES:
        raise ValueError("parent checkpoint is not the pilot-capped recipe")
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    sampler = payload.get("sampler")
    rng = payload.get("rng")
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
        raise ValueError("parent checkpoint lacks optimizer state")
    if not isinstance(scheduler, Mapping):
        raise ValueError("parent checkpoint lacks scheduler state")
    if not isinstance(sampler, Mapping):
        raise ValueError("parent checkpoint lacks sampler state")
    if not isinstance(rng, Mapping) or "torch" not in rng:
        raise ValueError("parent checkpoint lacks RNG state")
    steps = {
        int(state.get("step", -1))
        for state in optimizer["state"].values()
        if isinstance(state, Mapping)
    }
    if steps != {expected_updates}:
        raise ValueError("parent optimizer step counters are not exactly 50")
    if int(scheduler.get("last_epoch", -1)) != expected_updates:
        raise ValueError("parent scheduler is not at step 50")
    if int(sampler.get("cursor", -1)) < 0:
        raise ValueError("parent sampler cursor is invalid")
    model_identity = payload.get("model_identity")
    if not isinstance(model_identity, str) or not re.fullmatch(r"[0-9a-f]{64}", model_identity):
        raise ValueError("parent checkpoint model identity differs")
    return {
        "cumulative_updates": updates,
        "anchor_presentations": anchors,
        "next_optimizer_step": updates + 1,
        "optimizer_state_entries": len(optimizer["state"]),
        "optimizer_step_counters": sorted(steps),
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "sampler_cursor": int(sampler["cursor"]),
        "model_identity": model_identity,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_config_sha256": parent_config_sha256,
        "old_recipe_sha256": old_recipe_sha256,
    }


def validate_continuation_record(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a continuation checkpoint can be reloaded into the exact same state."""
    record = payload.get("continuation")
    if not isinstance(record, Mapping) or record.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("continuation checkpoint provenance schema differs")
    for key in (
        "new_recipe_sha256",
        "old_recipe_sha256",
        "parent_checkpoint_sha256",
        "parent_config_sha256",
        "migration_sha256",
        "authority_sha256",
    ):
        if record.get(key) != expected.get(key):
            raise ValueError(f"continuation provenance differs: {key}")
    if record.get("arm") != "C" or record.get("vision_unfreeze") is not False:
        raise ValueError("continuation checkpoint is not the frozen C arm")
    if int(record.get("validation_interval_updates", 0)) not in (100, 250):
        raise ValueError("continuation checkpoint validation interval differs")
    if int(record.get("recovery_checkpoints", 0)) != 2:
        raise ValueError("continuation checkpoint recovery retention differs")
    if list(record.get("milestones") or []) != list(MILESTONE_STEPS):
        raise ValueError("continuation checkpoint milestone set differs")
    if record.get("new_stage_elapsed_excludes_parent") is not True:
        raise ValueError("continuation checkpoint stage timing would include the parent")
    for flag in (
        "optimizer_state_present",
        "scheduler_state_present",
        "sampler_state_present",
        "rng_state_present",
    ):
        if record.get(flag) is not True:
            raise ValueError(f"continuation checkpoint lacks {flag}")
    if record.get("selector_self_sha_embedded") is not False:
        raise ValueError("continuation checkpoint self-SHA flag differs")
    if int(record.get("parent_cumulative_updates", -1)) != PARENT_UPDATES:
        raise ValueError("continuation checkpoint parent update counter differs")
    if int(record.get("parent_anchor_presentations", -1)) != PARENT_ANCHORS:
        raise ValueError("continuation checkpoint parent anchor counter differs")
    updates = int(payload.get("cumulative_updates", -1))
    anchors = int(payload.get("anchor_presentations", -1))
    if updates < PARENT_UPDATES + 1:
        raise ValueError("continuation checkpoint is not beyond the parent state")
    if anchors != updates * 16:
        raise ValueError("continuation checkpoint anchor counter is inconsistent")
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    sampler = payload.get("sampler")
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
        raise ValueError("continuation checkpoint lacks optimizer state")
    if not optimizer["state"]:
        raise ValueError("continuation checkpoint optimizer state is empty")
    scheduler = scheduler if isinstance(scheduler, Mapping) else {}
    if int(scheduler.get("last_epoch", -1)) != updates:
        raise ValueError("continuation scheduler is not at the checkpoint step")
    if not isinstance(sampler, Mapping) or int(sampler.get("batch_size", -1)) != 16:
        raise ValueError("continuation sampler state differs")
    steps = {
        int(state.get("step", -1))
        for state in optimizer["state"].values()
        if isinstance(state, Mapping)
    }
    if steps != {updates}:
        raise ValueError("continuation optimizer step counters differ from cumulative updates")
    identity = payload.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("continuation checkpoint lacks the new run identity")
    if identity.get("research_config_sha256") != expected.get("new_recipe_sha256"):
        raise ValueError("continuation checkpoint new recipe identity differs")
    training = payload.get("training_config")
    if not isinstance(training, Mapping):
        raise ValueError("continuation checkpoint lacks the new training recipe")
    if int(training.get("pilot_max_updates_per_arm", -1)) != TOTAL_UPDATES:
        raise ValueError("continuation checkpoint recipe still carries the pilot cap")
    if not isinstance(payload.get("rng"), Mapping):
        raise ValueError("continuation checkpoint lacks RNG state")
    return {
        "cumulative_updates": updates,
        "anchor_presentations": anchors,
        "next_optimizer_step": updates + 1,
        "optimizer_step_counters": sorted(steps),
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "sampler_cursor": int(sampler.get("cursor", -1)),
    }


# --------------------------------------------------------------------------- #
# Budget/disk accounting, retention, sampler
# --------------------------------------------------------------------------- #
def owned_run_bytes(root: str | Path) -> int:
    """Owned output bytes: skip symlinks and deduplicate hardlinks by inode."""
    root_path = Path(root)
    if not root_path.exists():
        return 0
    total = 0
    seen: set[tuple[int, int]] = set()
    for directory, dirnames, filenames in os.walk(root_path, followlinks=False):
        current = Path(directory)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (int(stat.st_dev), int(stat.st_ino))
            if key in seen:
                continue
            seen.add(key)
            total += int(stat.st_size)
    return total


def check_output_budget(root: str | Path, *, reserve_bytes: int, cap_bytes: int) -> int:
    used = owned_run_bytes(root)
    if used + int(reserve_bytes) > int(cap_bytes):
        raise ValueError(
            f"owned output bytes {used} + reserve {reserve_bytes} exceed cap {cap_bytes}"
        )
    return used


def plan_checkpoint_retention(
    entries: Sequence[Mapping[str, Any]],
    *,
    current_step: int,
    best_step: int | None,
    final_step: int | None,
    milestones: Sequence[int] = MILESTONE_STEPS,
    keep_recovery: int = 2,
) -> dict[str, list[str]]:
    """Keep latest two recovery checkpoints, milestones, best and final.

    Only entries named ``checkpoint_<6 digits>.pt`` are prunable; aliases, lineage
    files and other agents' files are never part of this plan.
    """
    pattern = re.compile(r"^checkpoint_(\d{6})\.pt$")
    managed: list[tuple[str, int]] = []
    keep: set[str] = set()
    for entry in entries:
        name = str(entry.get("name", ""))
        match = pattern.fullmatch(name)
        if not match:
            continue
        step = int(match.group(1))
        managed.append((name, step))
        if step in set(int(item) for item in milestones):
            keep.add(name)
        if final_step is not None and step == int(final_step):
            keep.add(name)
        if best_step is not None and step == int(best_step):
            keep.add(name)
    ordered = sorted(managed, key=lambda item: item[1])
    for name, _step in ordered[-max(0, int(keep_recovery)) :]:
        keep.add(name)
    delete = [name for name, _step in ordered if name not in keep]
    return {"keep": sorted(keep), "delete": sorted(delete)}


def _sampler_candidates(data: Any, cache: Any) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    for source in data.manifest.sources_for_split("train"):
        source_id = str(source.source_demo_id)
        for endpoint in range(25, int(source.num_frames), 5):
            row = cache.read(source_id, endpoint)
            if int(row.endpoint) != endpoint:
                raise ValueError("cache row endpoint differs")
            candidates.append((source_id, endpoint))
    if not candidates:
        raise ValueError("cache contains no training anchors")
    return candidates


def _cache_for(config: Mapping[str, Any]) -> tuple[Any, Any]:
    from dataclasses import replace

    from owac.raw.teacher_data import TeacherData, load_source_manifest
    from owac.raw.temporal_distillation import (
        TemporalAssetSet,
        TemporalCache,
        temporal_cache_identity,
    )

    assets = TemporalAssetSet.from_config(config, repo_root=REPO_ROOT)
    assets = replace(assets, device=str((config.get("assets") or {}).get("device", "cpu")))
    manifest = load_source_manifest(
        assets.sources_path,
        repo_root=REPO_ROOT,
        strict=True,
        expected_splits={"train": 40, "validation": 5, "test": 5},
    )
    data = TeacherData(manifest, repo_root=REPO_ROOT)
    canonical = replace(
        assets,
        device=("cuda:0" if str(assets.device).split(":")[0] == "cuda" else assets.device),
    )
    cache = TemporalCache(
        Path(str((config.get("assets") or {})["cache_root"])),
        temporal_cache_identity(canonical, data),
    )
    return cache, data


def _run_identity(config: Mapping[str, Any], bundle: Any, cache_key: str) -> dict[str, Any]:
    return {
        "scope": "stem_plus_gate",
        "research_config_sha256": recipe_digest(config),
        "cache_identity_key": cache_key,
        "teacher_artifact_sha256": bundle.identity["artifact_sha256"],
        "backend_sha256": bundle.identity["backend_sha256"],
        "tokenizer_sha256": bundle.identity["tokenizer_sha256"],
        "stats_sha256": bundle.identity["stats_sha256"],
        "sources_sha256": bundle.identity["sources_sha256"],
        "source_revision": bundle.identity.get("source_revision"),
    }


PARENT_ASSET_IDENTITY_KEYS: tuple[tuple[str, str], ...] = (
    # Actual parent run_identity key -> bundle identity key.  The parent uses
    # ``teacher_artifact_sha256``; the bundle exposes ``artifact_sha256``.
    ("teacher_artifact_sha256", "artifact_sha256"),
    ("backend_sha256", "backend_sha256"),
    ("tokenizer_sha256", "tokenizer_sha256"),
    ("stats_sha256", "stats_sha256"),
    ("sources_sha256", "sources_sha256"),
)


def _assert_parent_asset_identity(
    parent_run_identity: Mapping[str, Any], bundle_identity: Mapping[str, Any]
) -> None:
    """Exact parent->bundle SHA equality with the real parent key names.

    This is deliberately an explicit mapping, not a same-key lookup: the actual
    C50 run_identity stores ``teacher_artifact_sha256``, while the teacher bundle
    stores ``artifact_sha256``.  Values must still match exactly; no fallback or
    tolerance is permitted.
    """
    if not isinstance(parent_run_identity, Mapping) or not isinstance(bundle_identity, Mapping):
        raise ValueError("parent/bundle identity must be mappings")
    for parent_key, bundle_key in PARENT_ASSET_IDENTITY_KEYS:
        expected = bundle_identity.get(bundle_key)
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"bundle identity is missing {bundle_key}")
        if parent_key not in parent_run_identity:
            raise ValueError(f"parent run_identity is missing {parent_key}")
        actual = parent_run_identity[parent_key]
        if actual != expected:
            raise ValueError(f"parent checkpoint identity differs: {parent_key}")


def _require_carried_receipt_identity(
    receipt: Mapping[str, Any], *, identity: Mapping[str, Any], cache_key: str, kind: str
) -> None:
    expected_protocol = {
        "parity": "stage0-teacher-tokens-through-bridge-v1",
        "resource": "temporal-fullprefix-unroll6-stride5-v1",
    }[kind]
    if receipt.get("protocol") != expected_protocol:
        raise ValueError(f"carried {kind} readiness protocol differs")
    if receipt.get("passed") is not True or int(receipt.get("optimizer_updates", -1)) != 0:
        raise ValueError(f"carried {kind} readiness is not a zero-update pass")
    for key, expected in (
        ("teacher_artifact_sha256", identity["artifact_sha256"]),
        ("backend_sha256", identity["backend_sha256"]),
        ("sources_sha256", identity["sources_sha256"]),
        ("cache_identity_key", cache_key),
    ):
        if receipt.get(key) != expected:
            raise ValueError(f"carried {kind} readiness identity differs: {key}")


def _carried_readiness_ready(
    migration: MigrationValidation, runtime: Mapping[str, Any]
) -> dict[str, bool]:
    """Validate the actual migrated readiness receipts before any update.

    The parent receipts are not rewritten or re-signed; their full model/data
    identity is checked against this run and their original recipe SHA is kept
    as provenance in the migration record.
    """
    kinds: set[str] = set()
    for entry in migration.carried_receipts:
        receipt = _load_json(entry["path"])
        _require_carried_receipt_identity(
            receipt,
            identity=runtime["identity"],
            cache_key=runtime["cache"].identity_key,
            kind=str(entry["kind"]),
        )
        kinds.add(str(entry["kind"]))
    if kinds != {"parity", "resource"}:
        raise ValueError("both parity and resource readiness receipts are required")
    return {"parity": True, "resource": True}


def _crosscheck_receipt_paths(
    migration: MigrationValidation,
    *,
    parity_receipt: str | Path | None,
    resource_receipt: str | Path | None,
) -> None:
    """Confirm optional supervisor-supplied receipt paths are the carried ones."""
    for kind, value in (("parity", parity_receipt), ("resource", resource_receipt)):
        if value is None:
            continue
        entry = next((item for item in migration.carried_receipts if item["kind"] == kind), None)
        if entry is None:
            raise ValueError(f"no carried {kind} readiness receipt to cross-check")
        actual = Path(value).resolve()
        if actual != Path(str(entry["path"])).resolve():
            raise ValueError(f"{kind} receipt path differs from the signed migration")
        if _sha256_file(actual) != entry["sha256"]:
            raise ValueError(f"{kind} receipt SHA differs from the signed migration")


def _make_optimizer_scheduler(student: Any, settings: Any) -> tuple[Any, Any]:
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("student has no trainable parameters")
    optimizer = AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, settings.warmup_updates)),
    )
    return optimizer, scheduler


def _sampler_for(candidates: Sequence[tuple[str, int]], *, settings: Any) -> Any:
    from owac.training.raw_temporal_distillation import TemporalAnchorSampler

    return TemporalAnchorSampler(
        candidates, seed=int(settings.seed), batch_size=int(settings.effective_batch)
    )


# --------------------------------------------------------------------------- #
# Status, heartbeat and checkpoint files
# --------------------------------------------------------------------------- #
class RunStatusWriter:
    def __init__(self, path: str | Path, base: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.value = dict(base)

    def update(self, **changes: Any) -> dict[str, Any]:
        self.value.update(changes)
        self.value["updated_at"] = _now_iso()
        _atomic_json(self.path, self.value)
        return dict(self.value)

    def finish(self, *, status: str, native_exit: int, error: str | None = None) -> dict[str, Any]:
        now = _now_iso()
        self.value.update(
            {
                "status": status,
                "phase": "finished",
                "native_exit": int(native_exit),
                "finished_at": now,
                "error": error,
            }
        )
        self.value["updated_at"] = now
        _atomic_json(self.path, self.value)
        return dict(self.value)


class HeartbeatWriter:
    def __init__(self, path: str | Path, *, phase: str, cumulative_updates: int) -> None:
        self.path = Path(path)
        self._phase = phase
        self._cumulative = int(cumulative_updates)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name="c-full-heartbeat", daemon=True)

    def _payload(self) -> dict[str, Any]:
        pid = os.getpid()
        return {
            "schema": HEARTBEAT_SCHEMA,
            "pid": pid,
            "pid_start_ticks": _proc_start_ticks(pid),
            "pgid": os.getpgid(pid),
            "seq": self._sequence,
            "updated_at": _now_iso(),
            "phase": self._phase,
            "cumulative_updates": self._cumulative,
        }

    def write(self) -> None:
        with self._lock:
            self._sequence += 1
            _atomic_json(self.path, self._payload())

    def update(self, *, phase: str | None = None, cumulative_updates: int | None = None) -> None:
        with self._lock:
            if phase is not None:
                self._phase = str(phase)
            if cumulative_updates is not None:
                self._cumulative = int(cumulative_updates)
        self.write()

    def start(self) -> None:
        self.write()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self.write()

    def _run(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self.write()
            except OSError:
                return


def _checkpoint_path(checkpoints: Path, step: int) -> Path:
    return checkpoints / f"checkpoint_{int(step):06d}.pt"


def _final_checkpoint_path(checkpoints: Path, step: int) -> Path:
    return checkpoints / f"checkpoint_final_{int(step):06d}.pt"


def _link_alias(target: Path, alias: Path) -> None:
    temporary = alias.with_name(f".{alias.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    os.link(target, temporary)
    os.replace(temporary, alias)
    _fsync_dir(alias.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _save_checkpoint(
    trainer: Any,
    path: Path,
    *,
    optimizer: Any,
    scheduler: Any,
    sampler: Any,
    elapsed_seconds: float,
) -> str:
    """Atomically replace the checkpoint with the current complete training state."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.run_elapsed_seconds = float(elapsed_seconds)
    payload = trainer.checkpoint_payload(
        optimizer=optimizer,
        scheduler=scheduler,
        sampler_state=sampler.state(),
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def _same_file(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except OSError:
        return False


def _update_latest_aliases(checkpoints: Path, checkpoint: Path) -> None:
    """latest -> newest, previous -> the prior latest inode (recovery latest2)."""
    latest = checkpoints / "latest.pt"
    previous = checkpoints / "previous.pt"
    if latest.is_file() and not _same_file(latest, checkpoint):
        _link_alias(latest, previous)
    _link_alias(checkpoint, latest)


def _best_checkpoint_path(
    paths: Mapping[str, Path],
    *,
    best_step: int | None,
    current_step: int | None = None,
    current_path: Path | None = None,
) -> Path | None:
    if best_step is None:
        return None
    if int(best_step) == PARENT_UPDATES:
        return paths["parent_copy"]
    checkpoints = paths["checkpoints"]
    candidate = checkpoints / f"checkpoint_{int(best_step):06d}.pt"
    if candidate.is_file():
        return candidate
    candidate = checkpoints / f"checkpoint_final_{int(best_step):06d}.pt"
    if candidate.is_file():
        return candidate
    if (
        current_step is not None
        and int(best_step) == int(current_step)
        and current_path is not None
    ):
        return current_path
    return None


def _continuation_record(
    *,
    migration: MigrationValidation,
    authority_sha256: str,
    interval: int,
    milestones: Sequence[int],
    wall_cap: int,
    output_cap: int,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "arm": "C",
        "vision_unfreeze": False,
        "new_recipe_sha256": migration.new_recipe_sha256,
        "old_recipe_sha256": migration.old_recipe_sha256,
        "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
        "parent_config_sha256": migration.parent_config_sha256,
        "migration_sha256": migration.manifest_sha256,
        "authority_sha256": authority_sha256,
        "parent_cumulative_updates": PARENT_UPDATES,
        "parent_anchor_presentations": PARENT_ANCHORS,
        "selection_metric": "validation_token_total",
        "validation_interval_updates": int(interval),
        "recovery_checkpoints": 2,
        "milestones": [int(item) for item in milestones],
        "new_stage_wall_cap_seconds": int(wall_cap),
        "new_output_cap_bytes": int(output_cap),
        "new_stage_elapsed_excludes_parent": True,
        "optimizer_state_present": True,
        "scheduler_state_present": True,
        "sampler_state_present": True,
        "rng_state_present": True,
        # A file cannot contain its own final SHA.  The checkpoint selector records
        # the real SHA of an earlier best; for a self-referential best it records
        # null and selection.json is the authoritative actual-file SHA.
        "selector_self_sha_embedded": False,
        "created_at": _now_iso(),
    }


def _continuation_trainer_class() -> Any:
    from owac.training.raw_temporal_distillation import TemporalDistillationTrainer

    class ContinuationTrainer(TemporalDistillationTrainer):
        def __init__(
            self, *args: Any, continuation_record: Mapping[str, Any] | None = None, **kwargs: Any
        ) -> None:
            super().__init__(*args, **kwargs)
            self.continuation_record = (
                None if continuation_record is None else dict(continuation_record)
            )

        def checkpoint_payload(self, **kwargs: Any) -> dict[str, Any]:
            payload = super().checkpoint_payload(**kwargs)
            if self.continuation_record is not None:
                payload["continuation"] = dict(self.continuation_record)
            return payload

    return ContinuationTrainer


def _checkpoint_size_reserve(parent_checkpoint: Path) -> int:
    size = parent_checkpoint.stat().st_size
    return int(size * CHECKPOINT_RESERVE_FRACTION) + 64 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Parent history / selector normalization
# --------------------------------------------------------------------------- #
def _normalize_parent_history(progress_path: Path, target: Path) -> list[dict[str, Any]]:
    records = _read_jsonl_tolerant(progress_path)
    updates = []
    for record in records:
        if record.get("event") != "update":
            continue
        cumulative = int(record.get("cumulative_updates", -1))
        if cumulative < 1:
            continue
        updates.append(
            {
                "schema": PARENT_HISTORY_SCHEMA,
                "event": "parent_update",
                "cumulative_updates": cumulative,
                "anchor_presentations": int(record.get("anchor_presentations", 0)),
                "feature_mse": record.get("feature_mse"),
                "feature_cosine": record.get("feature_cosine"),
                "feature_total": record.get("feature_total"),
                "learning_rate": record.get("learning_rate"),
                "timestamp": record.get("timestamp"),
                "gpu_allocated_bytes": record.get("gpu_allocated_bytes"),
                "gpu_reserved_bytes": record.get("gpu_reserved_bytes"),
            }
        )
    updates.sort(key=lambda item: item["cumulative_updates"])
    if len(updates) != PARENT_UPDATES or [item["cumulative_updates"] for item in updates] != list(
        range(1, PARENT_UPDATES + 1)
    ):
        raise ValueError("parent progress.jsonl does not contain the actual 50 updates")
    if any(item["anchor_presentations"] != 16 * item["cumulative_updates"] for item in updates):
        raise ValueError("parent progress anchor counters are inconsistent")
    _atomic_text(target, "".join(json.dumps(item, sort_keys=True) + "\n" for item in updates))
    return updates


def _legacy_selector(parent_selection_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    selected: Mapping[str, Any] = {}
    source = "checkpoint"
    if parent_selection_path.is_file():
        value = _load_json(parent_selection_path)
        if isinstance(value, Mapping):
            selected = value
            source = str(parent_selection_path)
    checkpoint_selector = payload.get("selector")
    if not selected and isinstance(checkpoint_selector, Mapping):
        selected = checkpoint_selector
    best_step = selected.get("best_step")
    best_value = selected.get("best_value")
    checkpoint_sha = selected.get("checkpoint_sha256")
    return {
        "schema": "owac.raw-stem-temporal.c-full.legacy-selector.v1",
        "cleared": True,
        "used_for_selection": False,
        "metric": "physical_first5_mae7",
        "best_step": None if best_step is None else int(best_step),
        "best_value": None if best_value is None else float(best_value),
        "checkpoint_path": selected.get("checkpoint_path"),
        "checkpoint_sha256": checkpoint_sha,
        "source": source,
        "note": (
            "pilot action-MAE selector preserved for provenance only; the continuation "
            "selects on held-out projected-token MSE+0.1*cosine and action MAE is diagnostic"
        ),
        "recorded_at": _now_iso(),
    }


def _load_parent_baseline(run_root: Path) -> dict[str, Any]:
    path = run_root / "parent-baseline.json"
    if not path.is_file():
        raise ValueError("parent-baseline.json is absent; run prepare first")
    value = _load_json(path)
    if not isinstance(value, Mapping) or value.get("schema") != PARENT_BASELINE_SCHEMA:
        raise ValueError("parent baseline schema differs")
    return dict(value)


def _selector_from_files(run_root: Path) -> Any:
    from owac.evaluation.raw_temporal_features import FeatureSelector

    selector = FeatureSelector()
    baseline = _load_parent_baseline(run_root)
    selector.consider(
        int(baseline["step"]),
        baseline["record"],
        checkpoint_path=str(baseline["checkpoint_path"]),
        checkpoint_sha256=str(baseline["checkpoint_sha256"]),
    )
    for record in _read_jsonl_tolerant(run_root / "validation.jsonl"):
        if record.get("event") != "validation":
            continue
        selector.consider(
            int(record["step"]),
            record,
            checkpoint_path=record.get("checkpoint_path"),
            checkpoint_sha256=record.get("checkpoint_sha256"),
        )
    return selector


def _selection_matches(selector: Any, sidecar: Mapping[str, Any]) -> bool:
    return (
        sidecar.get("best_step") == selector.best_step
        and sidecar.get("best_value") == selector.best_value
    )


# --------------------------------------------------------------------------- #
# Model/asset construction and strict load
# --------------------------------------------------------------------------- #
def _build_assets(config: Mapping[str, Any], *, device: str | None) -> dict[str, Any]:
    from dataclasses import replace

    from owac.raw.temporal_distillation import (
        TemporalAssetSet,
        TemporalPacketDataset,
        build_teacher_bundle,
        build_temporal_student,
    )

    assets = TemporalAssetSet.from_config(config, repo_root=REPO_ROOT)
    if device:
        assets = replace(assets, device=device)
    identity = assets.validate_identity()
    cache, data = _cache_for(
        {**dict(config), "assets": {**config["assets"], "device": assets.device}}
    )
    bundle = build_teacher_bundle(assets, device=device or assets.device)
    student, bridge, report = build_temporal_student(
        bundle,
        stem_seed=int((config.get("student") or {}).get("stem_seed", 101)),
        input_mode="rawvla_rgb3",
    )
    dataset = TemporalPacketDataset(data=data, cache=cache, student=student, arm="C")
    return {
        "assets": assets,
        "identity": identity,
        "cache": cache,
        "data": data,
        "bundle": bundle,
        "student": student,
        "bridge": bridge,
        "report": report,
        "dataset": dataset,
        "device": device or assets.device,
    }


def _trainer(
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    run_identity: Mapping[str, Any],
    arm: str = "C",
    continuation_record: Mapping[str, Any] | None = None,
) -> Any:
    import torch

    from owac.training.raw_temporal_distillation import TemporalTrainingConfig

    bundle = runtime["bundle"]
    trainer_class = _continuation_trainer_class()
    return trainer_class(
        student=runtime["student"],
        bridge=runtime["bridge"],
        config=TemporalTrainingConfig.from_mapping(config),
        authority=None,
        teacher_ready=False,
        source_split_ready=False,
        parity_ready=False,
        resource_ready=False,
        teacher=torch.nn.ModuleList([bundle.teacher.policy, bundle.teacher.frontend.model]),
        arm=arm,
        run_identity=dict(run_identity),
        continuation_record=continuation_record,
    )


def _optimizer_state_steps(optimizer: Any) -> set[int]:
    steps: set[int] = set()
    state = optimizer.state_dict().get("state", {})
    for value in state.values():
        if isinstance(value, Mapping) and "step" in value:
            steps.add(int(value["step"]))
    return steps


def _strict_parent_load(
    runtime: Mapping[str, Any],
    *,
    parent_config: Mapping[str, Any],
    new_config: Mapping[str, Any],
    parent_copy: Path,
    parent_sha: str,
    migration: MigrationValidation,
    continuation_record: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from owac.training.raw_temporal_distillation import (
        TemporalTrainingConfig,
    )

    if _sha256_file(parent_copy) != parent_sha:
        raise ValueError("lineage parent checkpoint SHA differs")
    payload = _load_parent_payload(parent_copy)
    audit = validate_parent_checkpoint(
        payload,
        parent_checkpoint_sha256=parent_sha,
        old_recipe_sha256=migration.old_recipe_sha256,
        parent_config_sha256=migration.parent_config_sha256,
    )
    cache = runtime["cache"]
    bundle = runtime["bundle"]
    parent_identity = _run_identity(parent_config, bundle, cache.identity_key)
    if dict(payload.get("run_identity") or {}) != parent_identity:
        raise ValueError("parent checkpoint run identity does not match the parent config/assets")
    parent_settings = TemporalTrainingConfig.from_mapping(parent_config)
    trainer = _trainer(
        runtime,
        parent_config,
        run_identity=parent_identity,
        continuation_record=continuation_record,
    )
    optimizer, scheduler = _make_optimizer_scheduler(runtime["student"], parent_settings)
    trainer.load_checkpoint(parent_copy, optimizer=optimizer, scheduler=scheduler)
    if (
        trainer.cumulative_updates != PARENT_UPDATES
        or trainer.anchor_presentations != PARENT_ANCHORS
    ):
        raise ValueError("parent strict load did not restore C50/800 counters")
    if _optimizer_state_steps(optimizer) != {PARENT_UPDATES}:
        raise ValueError("parent strict load did not restore optimizer step 50")
    if int(scheduler.state_dict().get("last_epoch", -1)) != PARENT_UPDATES:
        raise ValueError("parent strict load did not restore scheduler step 50")
    if not isinstance(payload.get("sampler"), Mapping):
        raise ValueError("parent checkpoint sampler state is absent")
    new_settings = TemporalTrainingConfig.from_mapping(new_config)
    # Only after the strict parent load do we migrate the operational recipe.
    trainer.config = new_settings
    trainer.run_identity = _run_identity(new_config, bundle, cache.identity_key)
    if trainer.run_identity["research_config_sha256"] != migration.new_recipe_sha256:
        raise ValueError("new run identity recipe differs from the migration")
    trainer.run_elapsed_seconds = 0.0
    from owac.training.raw_temporal_distillation import TemporalAnchorSampler

    sampler = TemporalAnchorSampler(
        _sampler_candidates(runtime["data"], cache),
        seed=int(new_settings.seed),
        batch_size=int(new_settings.effective_batch),
    )
    sampler.load_state(payload["sampler"])
    return trainer, optimizer, scheduler, sampler, audit


def _load_parent_payload(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def _strict_continuation_load(
    runtime: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    checkpoint: Path,
    migration: MigrationValidation,
    authority_sha: str,
    continuation_record: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from owac.training.raw_temporal_distillation import (
        TemporalAnchorSampler,
        TemporalTrainingConfig,
    )

    payload = _load_parent_payload(checkpoint)
    settings = TemporalTrainingConfig.from_mapping(config)
    identity = _run_identity(config, runtime["bundle"], runtime["cache"].identity_key)
    expected = {
        "new_recipe_sha256": migration.new_recipe_sha256,
        "old_recipe_sha256": migration.old_recipe_sha256,
        "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
        "parent_config_sha256": migration.parent_config_sha256,
        "migration_sha256": migration.manifest_sha256,
        "authority_sha256": authority_sha,
    }
    audit = validate_continuation_record(payload, expected=expected)
    trainer = _trainer(
        runtime, config, run_identity=identity, continuation_record=continuation_record
    )
    optimizer, scheduler = _make_optimizer_scheduler(runtime["student"], settings)
    trainer.load_checkpoint(checkpoint, optimizer=optimizer, scheduler=scheduler)
    if trainer.cumulative_updates != int(payload["cumulative_updates"]):
        raise ValueError("continuation strict load did not restore cumulative updates")
    if _optimizer_state_steps(optimizer) != {trainer.cumulative_updates}:
        raise ValueError("continuation strict load did not restore optimizer counters")
    sampler = TemporalAnchorSampler(
        _sampler_candidates(runtime["data"], runtime["cache"]),
        seed=int(settings.seed),
        batch_size=int(settings.effective_batch),
    )
    sampler.load_state(payload["sampler"])
    return trainer, optimizer, scheduler, sampler, audit


def _evaluate_tokens(runtime: Mapping[str, Any], settings: Any) -> Any:
    from owac.evaluation.raw_temporal_features import evaluate_feature_validation

    student = runtime["student"]
    was_training = student.training
    student.eval()
    try:
        record = evaluate_feature_validation(
            student,
            runtime["dataset"],
            runtime["cache"],
            runtime["data"],
            cosine_weight=float(settings.cosine_weight),
            expected_sources=5,
            expected_windows=102,
        )
    finally:
        student.train(was_training)
    return record


def _validation_payload(
    runtime: Mapping[str, Any],
    settings: Any,
    *,
    step: int,
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    record = _evaluate_tokens(runtime, settings)
    return {
        "schema": VALIDATION_SCHEMA,
        "event": "validation",
        "timestamp": _now_iso(),
        "step": int(step),
        "new_stage_step": int(step) - PARENT_UPDATES,
        "split": "validation",
        "sources": record.sources,
        "trajectories": record.trajectories,
        "windows": record.windows,
        "anchors": record.windows,
        "episode_resets": record.episode_resets,
        "valid_tokens": record.valid_tokens,
        "scalar_denominator": record.scalar_denominator,
        "feature_mse": record.feature_mse,
        "feature_cosine": record.feature_cosine,
        "feature_total": record.feature_total,
        "cosine_weight": record.cosine_weight,
        "per_source": {name: summary.to_payload() for name, summary in record.per_source.items()},
        "selection_metric": "validation_token_total",
        "selection_eligible": True,
        "is_best": False,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "test_used": False,
        "action_diagnostic": None,
        "record": record.to_payload(),
    }


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def _prepared_paths(run_root: Path) -> dict[str, Path]:
    return {
        "run_root": run_root,
        "lineage": run_root / "lineage",
        "checkpoints": run_root / "checkpoints",
        "acceptance": run_root / "acceptance",
        "harness": run_root / "harness",
        "prepared": run_root / "prepared.json",
        "meta": run_root / "run-meta.json",
        "status": run_root / "run-status.json",
        "heartbeat": run_root / "heartbeat.json",
        "updates": run_root / "updates.jsonl",
        "validation": run_root / "validation.jsonl",
        "selection": run_root / "selection.json",
        "parent_baseline": run_root / "parent-baseline.json",
        "parent_history": run_root / "parent-history.jsonl",
        "legacy_selector": run_root / "lineage" / "parent-selector.json",
        "migration": run_root / "lineage" / "migration.json",
        "parent_copy": run_root / "lineage" / "parent-C50.pt",
    }


_ACTIVE_PREPARE: tuple[RunStatusWriter, HeartbeatWriter, float, str] | None = None


def _begin_prepare_lifecycle(
    paths: Mapping[str, Path],
    *,
    migration: MigrationValidation,
    authority_sha: str,
    target_cumulative: int,
) -> tuple[RunStatusWriter, HeartbeatWriter]:
    """Publish the real PID and a live heartbeat before any asset/model load."""
    global _ACTIVE_PREPARE
    if _ACTIVE_PREPARE is not None:
        raise ValueError("another prepare lifecycle is already active")
    started_at = _now_iso()
    status = RunStatusWriter(
        paths["status"],
        _status_base(
            budget=ContinuationBudget(),
            migration=migration,
            authority_sha=authority_sha,
            settings=None,
            target_cumulative=int(target_cumulative),
        ),
    )
    status.update(
        status="preparing",
        phase="preparing",
        pid=os.getpid(),
        pid_start_ticks=_proc_start_ticks(os.getpid()),
        pgid=os.getpgid(0),
        started_at=started_at,
        finished_at=None,
        native_exit=None,
        error=None,
        new_stage_wall_seconds=0.0,
        prepare_started_at=started_at,
    )
    heartbeat = HeartbeatWriter(
        paths["heartbeat"], phase="preparing", cumulative_updates=PARENT_UPDATES
    )
    heartbeat.start()
    _ACTIVE_PREPARE = (status, heartbeat, time.perf_counter(), started_at)
    return status, heartbeat


def _fail_prepare_lifecycle(
    error: BaseException,
    *,
    prepare_started: float | None = None,
) -> None:
    """Durably record a pre-optimization failure instead of a stale preparing state."""
    global _ACTIVE_PREPARE
    active = _ACTIVE_PREPARE
    if active is None:
        return
    status, heartbeat, started, _started_at = active
    elapsed = (
        max(0.0, time.perf_counter() - started)
        if prepare_started is None
        else max(0.0, float(prepare_started))
    )
    try:
        status.update(
            status="error",
            phase="error",
            pid=os.getpid(),
            pid_start_ticks=_proc_start_ticks(os.getpid()),
            pgid=os.getpgid(0),
            native_exit=2,
            finished_at=_now_iso(),
            prepare_wall_seconds=elapsed,
            new_stage_wall_seconds=elapsed,
            error=f"{error.__class__.__name__}: {error}",
            error_class=error.__class__.__name__,
        )
    except Exception:
        pass
    finally:
        _ACTIVE_PREPARE = None
        try:
            heartbeat.stop()
        except Exception:
            pass


def _complete_prepare_lifecycle() -> None:
    global _ACTIVE_PREPARE
    active = _ACTIVE_PREPARE
    if active is None:
        return
    _status, heartbeat, _started, _started_at = active
    _ACTIVE_PREPARE = None
    heartbeat.stop()


def _status_base(
    *,
    budget: ContinuationBudget,
    migration: MigrationValidation,
    authority_sha: str,
    settings: Any,
    target_cumulative: int,
) -> dict[str, Any]:
    pid = os.getpid()
    return {
        "schema": RUN_STATUS_SCHEMA,
        "status": "preparing",
        "phase": "preparing",
        "pid": pid,
        "pid_start_ticks": _proc_start_ticks(pid),
        "pgid": os.getpgid(pid),
        "native_exit": None,
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "finished_at": None,
        "cumulative_updates": PARENT_UPDATES,
        "parent_cumulative_updates": PARENT_UPDATES,
        "new_stage_updates": 0,
        "target_cumulative_updates": int(target_cumulative),
        "new_stage_target_updates": max(0, int(target_cumulative) - PARENT_UPDATES),
        "stop_after_update": None,
        "anchor_presentations": PARENT_ANCHORS,
        "new_stage_anchor_presentations": 0,
        "new_stage_wall_seconds": 0.0,
        "new_stage_wall_cap_seconds": budget.max_new_wall_seconds,
        "output_bytes": 0,
        "output_cap_bytes": budget.max_new_output_bytes,
        "last_checkpoint": None,
        "last_validation": None,
        "best": None,
        "selection_metric": "validation_token_total",
        "error": None,
        "closed_loop": "not_run",
        "closed_loop_episodes": 0,
        "control_restored": False,
        "recipe_sha256": migration.new_recipe_sha256,
        "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
        "migration_sha256": migration.manifest_sha256,
        "authority_sha256": authority_sha,
        "teacher_provenance": "local author-architecture init0 frontend + actual LIBERO pi05",
        "raw_origin": "LIBERO RGB8 inverse pseudoRAW3; not native Bayer",
    }


def _authority_payload(
    path: Path, *, config: Mapping[str, Any], user_authorization: Path
) -> dict[str, Any]:
    from owac.training.raw_temporal_distillation import TemporalTrainingAuthority

    if not path.is_file():
        raise ValueError("new C-full authority receipt is absent")
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("authority receipt must be an object")
    if value.get("authorized") is not True:
        raise ValueError("authority receipt is not authorized")
    if value.get("scope") not in {"raw-stem-temporal", "raw-stem-temporal-c-full"}:
        raise ValueError("authority scope differs")
    if value.get("no_closed_loop") is not True:
        raise ValueError("authority must explicitly refuse closed loop")
    if value.get("arm") not in (None, "C"):
        raise ValueError("authority is not C-only")
    budget = budget_from_user_authorization(user_authorization)
    max_updates = int(value.get("max_updates", 0))
    max_anchors = int(value.get("max_anchors", 0))
    max_wall = int(value.get("max_wall_seconds", 0))
    max_output = int(value.get("max_output_bytes", 0))
    if not (PARENT_UPDATES < max_updates <= budget.total_updates):
        raise ValueError("authority max_updates is outside the C-full cumulative cap")
    if not (PARENT_ANCHORS < max_anchors <= budget.total_anchors):
        raise ValueError("authority max_anchors is outside the C-full cumulative cap")
    if not (0 < max_wall <= budget.max_new_wall_seconds):
        raise ValueError("authority wall cap exceeds the new-stage wall cap")
    if not (0 < max_output <= budget.max_new_output_bytes):
        raise ValueError("authority output cap exceeds the new-output cap")
    receipt_sha = _require_sha(value.get("receipt_sha256"), "authority receipt_sha256")
    if receipt_sha != _sha256_file(user_authorization):
        raise ValueError("authority receipt is not bound to the current user authorization")
    declared_user = value.get("user_authorization")
    if declared_user is not None:
        if Path(str(declared_user)).resolve() != user_authorization.resolve():
            raise ValueError("authority user_authorization path differs")
    _ = TemporalTrainingAuthority.from_mapping({**dict(value), "authority_path": str(path)})
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "payload": dict(value),
        "max_updates": max_updates,
        "max_anchors": max_anchors,
        "max_wall_seconds": max_wall,
        "max_output_bytes": max_output,
        "receipt_sha256": receipt_sha,
        "config_recipe_sha256": recipe_digest(config),
    }


def _assert_parent_baseline(record: Any, migration: MigrationValidation) -> dict[str, Any]:
    expected = migration.expected_parent_baseline
    if int(record.windows) != 102 or int(record.sources) != 5 or int(record.episode_resets) != 5:
        raise ValueError(
            "C50 baseline source/window/reset count differs from the fixed 102/5/5 contract"
        )
    if int(record.valid_tokens) != 52224:
        raise ValueError("C50 baseline valid token count differs from 52224")
    if float(record.scalar_denominator) != 106954752.0:
        raise ValueError("C50 baseline scalar denominator differs from 106954752")
    deltas = {}
    for name in ("feature_mse", "feature_cosine", "feature_total"):
        actual = float(getattr(record, name))
        reference = float(expected[name])
        delta = abs(actual - reference)
        if delta > migration.baseline_tolerance_abs:
            raise ValueError(
                f"C50 baseline {name} differs by {delta!r} > 1e-6 "
                f"(actual {actual!r}, attached-assessment {reference!r}); "
                "locate the numerical cause, do not loosen the tolerance"
            )
        deltas[name] = {"actual": actual, "reference": reference, "abs_delta": delta}
    deltas["tolerance_abs"] = migration.baseline_tolerance_abs
    deltas["source"] = str(migration.parent_assessment)
    deltas["source_sha256"] = migration.parent_assessment_sha256
    return deltas


def prepare(
    *,
    config_path: str | Path,
    migration_path: str | Path,
    authority_path: str | Path,
    parity_receipt: str | Path | None = None,
    resource_receipt: str | Path | None = None,
    device: str | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    prepare_started = time.perf_counter()
    config = _load_json(config_path)
    if not isinstance(config, Mapping):
        raise ValueError("config must be an object")
    migration = validate_migration_manifest(
        migration_path,
        new_config_path=config_path,
        parent_config_path=REPO_ROOT
        / "outputs/raw-distillation-pilot-20261021-101725/authorized-config.json",
        parent_checkpoint_path=REPO_ROOT
        / "outputs/raw-distillation-pilot-20261021-101725/pilot-C/checkpoint_final_000050.pt",
        user_authorization_path=REPO_ROOT
        / "outputs/raw-distillation-c-full-20261021-123313/user-authorization.json",
        run_root=run_root,
    )
    authority = _authority_payload(
        Path(authority_path),
        config=config,
        user_authorization=migration.user_authorization
        if migration.user_authorization.is_absolute()
        else REPO_ROOT / migration.user_authorization,
    )
    _crosscheck_receipt_paths(
        migration,
        parity_receipt=parity_receipt,
        resource_receipt=resource_receipt,
    )
    root = migration.run_root
    paths = _prepared_paths(root)
    if paths["prepared"].is_file():
        existing = _load_json(paths["prepared"])
        if (
            existing.get("new_recipe_sha256") == migration.new_recipe_sha256
            and existing.get("parent_checkpoint_sha256") == migration.parent_checkpoint_sha256
            and existing.get("migration_sha256") == migration.manifest_sha256
            and existing.get("authority_sha256") == authority["sha256"]
        ):
            return {
                "status": "already_prepared",
                "run_root": str(root),
                "prepared": str(paths["prepared"]),
                "cumulative_updates": PARENT_UPDATES,
                "next_optimizer_step": PARENT_UPDATES + 1,
                "optimizer_updates": 0,
                "prepare_wall_seconds": existing.get("prepare_wall_seconds"),
            }
        raise ValueError("run root already has a different prepared receipt")
    for directory in ("lineage", "checkpoints", "acceptance"):
        paths[directory].mkdir(parents=True, exist_ok=True)
    target_cap = min(
        TOTAL_UPDATES,
        int(authority["max_updates"]),
        int((config.get("training") or {}).get("max_cumulative_updates_per_arm", TOTAL_UPDATES)),
    )
    status, heartbeat = _begin_prepare_lifecycle(
        paths,
        migration=migration,
        authority_sha=authority["sha256"],
        target_cumulative=target_cap,
    )
    reserve = _checkpoint_size_reserve(migration.parent_checkpoint)
    used = check_output_budget(
        root, reserve_bytes=reserve, cap_bytes=ContinuationBudget().max_new_output_bytes
    )
    if not paths["parent_copy"].is_file():
        if _sha256_file(migration.parent_checkpoint) != migration.parent_checkpoint_sha256:
            raise ValueError("parent checkpoint SHA changed since migration validation")
        temporary = paths["parent_copy"].with_name(f".{paths['parent_copy'].name}.tmp")
        shutil.copyfile(migration.parent_checkpoint, temporary)
        _fsync_file(temporary)
        os.replace(temporary, paths["parent_copy"])
        _fsync_dir(paths["lineage"])
    if _sha256_file(paths["parent_copy"]) != migration.parent_checkpoint_sha256:
        raise ValueError("lineage parent copy SHA differs")
    parent_payload = _load_parent_payload(paths["parent_copy"])

    runtime = _build_assets(config, device=device)
    identity = runtime["identity"]
    for entry in migration.carried_receipts:
        receipt = _load_json(entry["path"])
        _require_carried_receipt_identity(
            receipt,
            identity=identity,
            cache_key=runtime["cache"].identity_key,
            kind=str(entry["kind"]),
        )
    if runtime["cache"].identity_key != parent_payload["run_identity"]["cache_identity_key"]:
        raise ValueError("cache identity differs from the parent checkpoint")
    _assert_parent_asset_identity(parent_payload["run_identity"], identity)

    parent_config = _load_json(migration.parent_config)
    trainer, optimizer, scheduler, sampler, parent_audit = _strict_parent_load(
        runtime,
        parent_config=parent_config,
        new_config=config,
        parent_copy=paths["parent_copy"],
        parent_sha=migration.parent_checkpoint_sha256,
        migration=migration,
    )
    settings = trainer.config
    # Clear the old action-MAE selector only after preserving its provenance.
    parent_selection_path = (
        REPO_ROOT / "outputs/raw-distillation-pilot-20261021-101725/pilot-C/selection.json"
    )
    legacy = _legacy_selector(parent_selection_path, parent_payload)
    _atomic_json(paths["legacy_selector"], legacy)
    _normalize_parent_history(
        REPO_ROOT / "outputs/raw-distillation-pilot-20261021-101725/pilot-C/progress.jsonl",
        paths["parent_history"],
    )
    baseline_record = _evaluate_tokens(runtime, settings)
    baseline_deltas = _assert_parent_baseline(baseline_record, migration)
    baseline = {
        "schema": PARENT_BASELINE_SCHEMA,
        "step": PARENT_UPDATES,
        "checkpoint_path": str(paths["parent_copy"].relative_to(root)),
        "checkpoint_sha256": migration.parent_checkpoint_sha256,
        "created_at": _now_iso(),
        "feature_mse": baseline_record.feature_mse,
        "feature_cosine": baseline_record.feature_cosine,
        "feature_total": baseline_record.feature_total,
        "valid_tokens": baseline_record.valid_tokens,
        "scalar_denominator": baseline_record.scalar_denominator,
        "windows": baseline_record.windows,
        "sources": baseline_record.sources,
        "episode_resets": baseline_record.episode_resets,
        "record": baseline_record.to_payload(),
        "attached_assessment_check": baseline_deltas,
    }
    _atomic_json(paths["parent_baseline"], baseline)
    trainer.set_selector_state(
        best_value=float(baseline_record.feature_total),
        best_step=PARENT_UPDATES,
        best_checkpoint_path=str(paths["parent_copy"]),
        best_checkpoint_sha256=migration.parent_checkpoint_sha256,
        best_metrics=baseline["record"],
    )
    selector = _selector_from_files(root)
    selection = selector.sidecar(
        checkpoint_path=str(paths["parent_copy"]),
        checkpoint_sha256=migration.parent_checkpoint_sha256,
        parent_baseline=baseline,
        legacy_action_selector=legacy,
        updated_at=_now_iso(),
    )
    _atomic_json(paths["selection"], selection)
    status.update(
        status="preparing",
        phase="preparing",
        cumulative_updates=PARENT_UPDATES,
        anchor_presentations=PARENT_ANCHORS,
        output_bytes=owned_run_bytes(root),
        last_checkpoint={
            "step": PARENT_UPDATES,
            "path": str(paths["parent_copy"].relative_to(root)),
            "sha256": migration.parent_checkpoint_sha256,
            "kind": "parent",
        },
        last_validation={
            "step": PARENT_UPDATES,
            "checkpoint_sha256": migration.parent_checkpoint_sha256,
            "feature_mse": baseline_record.feature_mse,
            "feature_cosine": baseline_record.feature_cosine,
            "feature_total": baseline_record.feature_total,
            "valid_tokens": baseline_record.valid_tokens,
            "windows": baseline_record.windows,
            "sources": baseline_record.sources,
        },
        best={
            "step": PARENT_UPDATES,
            "value": baseline_record.feature_total,
            "checkpoint_sha256": migration.parent_checkpoint_sha256,
            "metric": "validation_token_total",
        },
    )
    meta = {
        "schema": RUN_META_SCHEMA,
        "created_at": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "git_head": _git_head(),
        "run_root": str(root),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": migration.new_config_sha256,
        "new_recipe_sha256": migration.new_recipe_sha256,
        "parent_config_path": str(migration.parent_config),
        "parent_config_sha256": migration.parent_config_sha256,
        "old_recipe_sha256": migration.old_recipe_sha256,
        "parent_checkpoint_path": str(migration.parent_checkpoint),
        "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
        "parent_audit": parent_audit,
        "migration_path": str(migration.manifest_path),
        "migration_sha256": migration.manifest_sha256,
        "allowed_operational_changes": list(migration.allowed_operational_changes),
        "authority_path": authority["path"],
        "authority_sha256": authority["sha256"],
        "authority_receipt_sha256": authority["receipt_sha256"],
        "carried_readiness_receipts": list(migration.carried_receipts),
        "budget": asdict(ContinuationBudget()),
        "selection_metric": "validation_token_total",
        "validation_interval_updates": VALIDATION_INTERVAL,
        "milestone_steps": list(MILESTONE_STEPS),
        "teacher_provenance": "local author-architecture init0 frontend + actual LIBERO pi05",
        "raw_origin": "LIBERO RGB8 inverse pseudoRAW3; not native Bayer",
        "closed_loop": "not_run",
        "control_restored": False,
        "student_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in runtime["student"].parameters()
                if parameter.requires_grad
            )
        ),
        "optimizer_updates_performed": 0,
        "checkpoint_retention": {
            "interval": int((config.get("training") or {}).get("checkpoint_interval_updates", 250)),
            "recovery_checkpoints": int(
                (config.get("training") or {}).get("recovery_checkpoints", 2)
            ),
            "milestones": list(MILESTONE_STEPS),
            "best_pinned": True,
        },
        "teacher_rgb_used_for_c": False,
        "code_sha256": {
            "continuation": _sha256_file(Path(__file__)),
            "features": _sha256_file(REPO_ROOT / "src/owac/evaluation/raw_temporal_features.py"),
        },
        "output_bytes_at_prepare": owned_run_bytes(root),
        "parent_bytes_seen": used,
        "prepare_wall_seconds": time.perf_counter() - prepare_started,
    }
    _atomic_json(paths["meta"], meta)
    prepared = {
        "schema": PREPARED_SCHEMA,
        "prepared_at": _now_iso(),
        "run_root": str(root),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": migration.new_config_sha256,
        "new_recipe_sha256": migration.new_recipe_sha256,
        "old_recipe_sha256": migration.old_recipe_sha256,
        "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
        "parent_config_sha256": migration.parent_config_sha256,
        "migration_sha256": migration.manifest_sha256,
        "authority_sha256": authority["sha256"],
        "parent_audit": parent_audit,
        "baseline": {
            "feature_mse": baseline_record.feature_mse,
            "feature_cosine": baseline_record.feature_cosine,
            "feature_total": baseline_record.feature_total,
            "valid_tokens": baseline_record.valid_tokens,
            "scalar_denominator": baseline_record.scalar_denominator,
            "windows": baseline_record.windows,
            "sources": baseline_record.sources,
            "episode_resets": baseline_record.episode_resets,
        },
        "optimizer_updates": 0,
        "next_optimizer_step": PARENT_UPDATES + 1,
        "prepare_wall_seconds": time.perf_counter() - prepare_started,
        "closed_loop": "not_run",
        "control_restored": False,
    }
    _atomic_json(paths["prepared"], prepared)
    status.update(
        status="prepared",
        phase="preparing",
        prepare_wall_seconds=prepared["prepare_wall_seconds"],
        new_stage_wall_seconds=prepared["prepare_wall_seconds"],
        finished_at=_now_iso(),
        native_exit=0,
        error=None,
        output_bytes=owned_run_bytes(root),
        last_checkpoint={
            "step": PARENT_UPDATES,
            "path": str(paths["parent_copy"].relative_to(root)),
            "sha256": migration.parent_checkpoint_sha256,
            "kind": "parent",
        },
    )
    _complete_prepare_lifecycle()
    return {
        "status": "prepared",
        "run_root": str(root),
        "cumulative_updates": PARENT_UPDATES,
        "next_optimizer_step": PARENT_UPDATES + 1,
        "optimizer_updates": 0,
        "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
        "new_recipe_sha256": migration.new_recipe_sha256,
        "baseline_feature_mse": baseline_record.feature_mse,
        "baseline_feature_cosine": baseline_record.feature_cosine,
        "baseline_feature_total": baseline_record.feature_total,
        "validation_windows": baseline_record.windows,
        "validation_sources": baseline_record.sources,
        "carried_receipts": [entry["kind"] for entry in migration.carried_receipts],
        "prepare_wall_seconds": prepared["prepare_wall_seconds"],
        "closed_loop": "not_run",
        "control_restored": False,
    }


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _phase_acceptance(run_root: Path, *, step: int, checkpoint_sha256: str) -> dict[str, Any]:
    path = run_root / "acceptance" / f"phase-{int(step):04d}.json"
    if not path.is_file():
        raise ValueError("phase acceptance receipt is absent")
    value = _load_json(path)
    if not isinstance(value, Mapping) or value.get("schema") != (
        "owac.raw-stem-temporal.c-full.phase-acceptance.v1"
    ):
        raise ValueError("phase acceptance schema differs")
    if value.get("accepted") is not True:
        raise ValueError("phase acceptance is not accepted")
    if int(value.get("step", -1)) != int(step):
        raise ValueError("phase acceptance step differs")
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("phase acceptance checkpoint SHA differs")
    if not str(value.get("reviewer") or "").strip():
        raise ValueError("phase acceptance needs a reviewer")
    return dict(value)


def _resume_payload_for_train(
    runtime: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    migration: MigrationValidation,
    authority_sha: str,
    resume: Path,
    stop_after: int,
    continuation_record: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    payload = _load_parent_payload(resume)
    if payload.get("schema") == "owac.raw-stem-temporal.checkpoint.v1":
        # The actual C50 parent handed in directly still goes through the narrow
        # migration; a continuation checkpoint uses its new recipe/provenance.
        if int(payload.get("cumulative_updates", -1)) == PARENT_UPDATES:
            parent_config = _load_json(migration.parent_config)
            return _strict_parent_load(
                runtime,
                parent_config=parent_config,
                new_config=config,
                parent_copy=resume,
                parent_sha=migration.parent_checkpoint_sha256,
                migration=migration,
                continuation_record=continuation_record,
            )
        trainer, optimizer, scheduler, sampler, audit = _strict_continuation_load(
            runtime,
            config=config,
            checkpoint=resume,
            migration=migration,
            authority_sha=authority_sha,
            continuation_record=continuation_record,
        )
        if trainer.cumulative_updates == PARENT_UPDATES + 1 and stop_after > PARENT_UPDATES + 1:
            _phase_acceptance(
                migration.run_root,
                step=PARENT_UPDATES + 1,
                checkpoint_sha256=_sha256_file(resume),
            )
        return trainer, optimizer, scheduler, sampler, audit
    raise ValueError("resume checkpoint schema differs")


def _status_from_run(run_root: Path) -> dict[str, Any]:
    path = run_root / "run-status.json"
    if not path.is_file():
        return {"status": "unknown", "run_root": str(run_root)}
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError("run status is not an object")
    return value


def train(
    *,
    config_path: str | Path,
    migration_path: str | Path,
    authority_path: str | Path,
    parity_receipt: str | Path | None = None,
    resource_receipt: str | Path | None = None,
    stop_after_update: int | None = None,
    resume: str | Path | None = None,
    acceptance: str | Path | None = None,
    device: str | None = None,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    from owac.training.raw_temporal_distillation import TemporalTrainingConfig

    config = _load_json(config_path)
    migration = validate_migration_manifest(
        migration_path,
        new_config_path=config_path,
        parent_config_path=REPO_ROOT
        / "outputs/raw-distillation-pilot-20261021-101725/authorized-config.json",
        parent_checkpoint_path=REPO_ROOT
        / "outputs/raw-distillation-pilot-20261021-101725/pilot-C/checkpoint_final_000050.pt",
        user_authorization_path=REPO_ROOT
        / "outputs/raw-distillation-c-full-20261021-123313/user-authorization.json",
        run_root=run_root,
    )
    authority = _authority_payload(
        Path(authority_path),
        config=config,
        user_authorization=migration.user_authorization,
    )
    _crosscheck_receipt_paths(
        migration,
        parity_receipt=parity_receipt,
        resource_receipt=resource_receipt,
    )
    root = migration.run_root
    paths = _prepared_paths(root)
    prepared = _load_json(paths["prepared"]) if paths["prepared"].is_file() else None
    if not isinstance(prepared, Mapping):
        raise ValueError("run is not prepared; run prepare first")
    for key in (
        "new_recipe_sha256",
        "parent_checkpoint_sha256",
        "parent_config_sha256",
        "migration_sha256",
        "authority_sha256",
    ):
        expected = {
            "new_recipe_sha256": migration.new_recipe_sha256,
            "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
            "parent_config_sha256": migration.parent_config_sha256,
            "migration_sha256": migration.manifest_sha256,
            "authority_sha256": authority["sha256"],
        }[key]
        if prepared.get(key) != expected:
            raise ValueError(f"prepared receipt differs: {key}")
    baseline = _load_parent_baseline(root)
    runtime = _build_assets(config, device=device)
    readiness = _carried_readiness_ready(migration, runtime)
    settings = TemporalTrainingConfig.from_mapping(config)
    training_mapping = dict(config.get("training") or {})
    wall_cap = int(training_mapping.get("max_wall_seconds", 0))
    output_cap = int(training_mapping.get("max_output_bytes", 0))
    interval = int(training_mapping.get("checkpoint_interval_updates", 250))
    if wall_cap <= 0 or output_cap <= 0:
        raise ValueError("new config must declare positive wall/output caps")
    if interval not in (100, 250):
        raise ValueError("checkpoint interval must be 100 or 250")
    continuation_record = _continuation_record(
        migration=migration,
        authority_sha256=authority["sha256"],
        interval=interval,
        milestones=MILESTONE_STEPS,
        wall_cap=wall_cap,
        output_cap=output_cap,
    )
    cap = min(TOTAL_UPDATES, authority["max_updates"], settings.max_cumulative_updates_per_arm)
    stop_target = cap if stop_after_update is None else int(stop_after_update)
    if stop_target <= PARENT_UPDATES or stop_target > cap:
        raise ValueError("stop-after-update must be in (50, declared cap]")
    resume_path = Path(resume).resolve() if resume is not None else paths["parent_copy"]
    trainer, optimizer, scheduler, sampler, load_audit = _resume_payload_for_train(
        runtime,
        config=config,
        migration=migration,
        authority_sha=authority["sha256"],
        resume=resume_path,
        stop_after=stop_target,
        continuation_record=continuation_record,
    )
    if trainer.continuation_record is None:
        trainer.continuation_record = dict(continuation_record)
    else:
        actual_record = dict(trainer.continuation_record)
        actual_record.pop("created_at", None)
        expected_record = dict(continuation_record)
        expected_record.pop("created_at", None)
        if actual_record != expected_record:
            raise ValueError("continuation checkpoint provenance differs from the new recipe")
    from owac.training.raw_temporal_distillation import TemporalTrainingAuthority

    trainer.authority = TemporalTrainingAuthority.from_mapping(
        {**authority["payload"], "authority_path": authority["path"]}
    )
    trainer.teacher_ready = True
    trainer.source_split_ready = True
    trainer.parity_ready = bool(readiness["parity"])
    trainer.resource_ready = bool(readiness["resource"])
    if not (trainer.parity_ready and trainer.resource_ready):
        raise ValueError("migrated readiness receipts are not ready")
    if trainer.cumulative_updates >= stop_target:
        return {
            "status": "no_op",
            "reason": "checkpoint is already at or beyond stop-after-update",
            "cumulative_updates": trainer.cumulative_updates,
            "stop_after_update": stop_target,
        }
    if acceptance is not None:
        path = Path(acceptance).resolve()
        value = _load_json(path)
        if value.get("accepted") is not True:
            raise ValueError("provided acceptance receipt is not accepted")
        if int(value.get("step", -1)) != trainer.cumulative_updates:
            raise ValueError("provided acceptance step differs from the resume checkpoint")
        if value.get("checkpoint_sha256") != _sha256_file(resume_path):
            raise ValueError("provided acceptance checkpoint SHA differs")
    elif trainer.cumulative_updates == PARENT_UPDATES + 1 and stop_target > PARENT_UPDATES + 1:
        _phase_acceptance(
            root,
            step=PARENT_UPDATES + 1,
            checkpoint_sha256=_sha256_file(resume_path),
        )
    selector = _selector_from_files(root)
    sidecar_path = paths["selection"]
    if sidecar_path.is_file() and not _selection_matches(selector, _load_json(sidecar_path)):
        raise ValueError("selection sidecar does not match the reconstructed token selector")
    if selector.best_step is None:
        selector.consider(
            PARENT_UPDATES,
            baseline["record"],
            checkpoint_path=baseline["checkpoint_path"],
            checkpoint_sha256=baseline["checkpoint_sha256"],
        )
    reserve = _checkpoint_size_reserve(migration.parent_checkpoint)
    used = check_output_budget(root, reserve_bytes=reserve, cap_bytes=output_cap)
    stage_wall_before = float(trainer.run_elapsed_seconds)
    run_started = time.perf_counter() - stage_wall_before
    base_status = (
        _status_from_run(root)
        if paths["status"].is_file()
        else _status_base(
            budget=ContinuationBudget(),
            migration=migration,
            authority_sha=authority["sha256"],
            settings=settings,
            target_cumulative=cap,
        )
    )
    base_status.update(
        {
            "status": "running",
            "phase": "running",
            "pid": os.getpid(),
            "pid_start_ticks": _proc_start_ticks(os.getpid()),
            "pgid": os.getpgid(os.getpid()),
            "native_exit": None,
            "started_at": _now_iso(),
            "finished_at": None,
            "target_cumulative_updates": cap,
            "new_stage_target_updates": max(0, cap - PARENT_UPDATES),
            "parent_cumulative_updates": PARENT_UPDATES,
            "stop_after_update": stop_target,
            "cumulative_updates": trainer.cumulative_updates,
            "anchor_presentations": trainer.anchor_presentations,
            "new_stage_updates": trainer.cumulative_updates - PARENT_UPDATES,
            "new_stage_anchor_presentations": trainer.anchor_presentations - PARENT_ANCHORS,
            "output_bytes": used,
            "output_cap_bytes": output_cap,
            "new_stage_wall_cap_seconds": wall_cap,
            "recipe_sha256": migration.new_recipe_sha256,
            "authority_sha256": authority["sha256"],
            "selection_metric": "validation_token_total",
            "stop_reason": None,
        }
    )
    status = RunStatusWriter(paths["status"], base_status)
    heartbeat = HeartbeatWriter(
        paths["heartbeat"], phase="running", cumulative_updates=trainer.cumulative_updates
    )
    stop_requested = {"value": False}

    def _request_stop(_signum: int, _frame: Any) -> None:
        stop_requested["value"] = True

    previous_term = signal.signal(signal.SIGTERM, _request_stop)
    previous_int = signal.signal(signal.SIGINT, _request_stop)
    heartbeat.start()
    updates = int(trainer.cumulative_updates)
    new_stage_start_updates = updates
    resource_blocked: str | None = None
    earlystop = False
    last_validation: dict[str, Any] | None = None
    last_checkpoint: dict[str, Any] | None = None
    validated_steps: set[int] = {PARENT_UPDATES}
    for record in _read_jsonl_tolerant(paths["validation"]):
        if record.get("event") == "validation":
            validated_steps.add(int(record["step"]))
    hard_anchors = min(
        TOTAL_ANCHORS, authority["max_anchors"], settings.max_anchor_presentations_per_arm
    )
    validated_records: dict[int, dict[str, Any]] = {}
    for existing in _read_jsonl_tolerant(paths["validation"]):
        if existing.get("event") == "validation":
            validated_records[int(existing["step"])] = existing
    validated_steps: set[int] = set(validated_records) | {PARENT_UPDATES}

    def _relative(path: Path) -> str:
        return str(path.relative_to(root))

    def _best_sha(
        best_path: Path | None, *, current_path: Path | None, current_sha: str | None
    ) -> str | None:
        if best_path is None:
            return None
        if best_path == paths["parent_copy"]:
            return migration.parent_checkpoint_sha256
        if current_path is not None and best_path == current_path:
            return current_sha
        return _sha256_file(best_path) if best_path.is_file() else None

    def _publish_selection() -> dict[str, Any]:
        best_path = _best_checkpoint_path(
            paths,
            best_step=selector.best_step,
            current_step=updates,
            current_path=last_checkpoint_path,
        )
        if best_path is None:
            raise ValueError("selected token checkpoint is absent")
        if best_path == paths["parent_copy"]:
            best_sha = migration.parent_checkpoint_sha256
        else:
            if not best_path.is_file():
                raise ValueError("selected token checkpoint file is absent")
            best_sha = _sha256_file(best_path)
        _link_alias(best_path, paths["checkpoints"] / "best.pt")
        sidecar = selector.sidecar(
            checkpoint_path=_relative(best_path),
            checkpoint_sha256=best_sha,
            parent_baseline=baseline,
            legacy_action_selector=_load_json(paths["legacy_selector"]),
            updated_at=_now_iso(),
        )
        _atomic_json(paths["selection"], sidecar)
        return sidecar

    def _persist_point(step: int, *, final: bool, allow_validation: bool) -> str:
        nonlocal last_validation, last_checkpoint, last_checkpoint_path
        stage_wall = time.perf_counter() - run_started
        used = owned_run_bytes(root)
        if used + 2 * reserve > output_cap:
            raise _ResourceStop("output_cap")
        checkpoint = (
            _final_checkpoint_path(paths["checkpoints"], step)
            if final
            else _checkpoint_path(paths["checkpoints"], step)
        )
        relative = _relative(checkpoint)
        validation = validated_records.get(step)
        if validation is None and allow_validation:
            validation = _validation_payload(
                runtime,
                settings,
                step=step,
                checkpoint_path=relative,
                checkpoint_sha256=None,
            )
            improved = selector.consider(
                step,
                validation,
                checkpoint_path=relative,
                checkpoint_sha256=None,
            )
            validation["is_best"] = bool(improved)
        # Selector is updated BEFORE the authoritative atomic save.  If this
        # checkpoint is itself the new best, its own final file SHA cannot be
        # embedded in the same file (self-reference); selection.json carries the
        # actual SHA and the checkpoint records null for this self case.
        best_path = _best_checkpoint_path(
            paths,
            best_step=selector.best_step,
            current_step=step,
            current_path=checkpoint,
        )
        best_existing_sha = _best_sha(best_path, current_path=checkpoint, current_sha=None)
        # The checkpoint selector must carry the metrics of the actual best step,
        # not the current (possibly worse) validation point.
        if selector.best_record is None:
            raise ValueError("token selector has no record for its best step")
        if int(selector.best_record.get("step", -1)) != int(selector.best_step):
            raise ValueError("token selector best metrics belong to a different step")
        trainer.set_selector_state(
            best_value=float(selector.best_value),
            best_step=int(selector.best_step),
            best_checkpoint_path=(_relative(best_path) if best_path is not None else relative),
            best_checkpoint_sha256=best_existing_sha,
            best_metrics=dict(selector.best_record),
        )
        sha = _save_checkpoint(
            trainer,
            checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            elapsed_seconds=stage_wall,
        )
        last_checkpoint = {
            "step": int(step),
            "path": relative,
            "sha256": sha,
            "kind": "final" if final else ("milestone" if step in MILESTONE_STEPS else "interval"),
        }
        last_checkpoint_path = checkpoint
        _update_latest_aliases(paths["checkpoints"], checkpoint)
        if validation is not None:
            validation["checkpoint_sha256"] = sha
            validation["checkpoint_path"] = relative
            validation["action_diagnostic"] = None
            if step not in validated_records:
                _append_jsonl(paths["validation"], validation)
                validated_records[int(step)] = validation
                validated_steps.add(int(step))
            last_validation = validation
        best_path = _best_checkpoint_path(
            paths,
            best_step=selector.best_step,
            current_step=step,
            current_path=checkpoint,
        )
        best_sha = _best_sha(best_path, current_path=checkpoint, current_sha=sha)
        if best_sha is None:
            raise ValueError("selected checkpoint has no actual SHA")
        trainer.selector_state["best_checkpoint_sha256"] = best_sha
        selection = _publish_selection()
        _prune_checkpoints(
            paths["checkpoints"],
            current_step=step,
            best_step=int(selector.best_step),
            final_step=step if final else None,
        )
        status.update(
            status="running",
            phase="running",
            cumulative_updates=step,
            anchor_presentations=trainer.anchor_presentations,
            new_stage_updates=step - PARENT_UPDATES,
            new_stage_anchor_presentations=trainer.anchor_presentations - PARENT_ANCHORS,
            new_stage_wall_seconds=stage_wall,
            output_bytes=owned_run_bytes(root),
            last_checkpoint=last_checkpoint,
            last_validation=(
                None
                if last_validation is None
                else {
                    key: last_validation[key]
                    for key in (
                        "step",
                        "checkpoint_sha256",
                        "feature_mse",
                        "feature_cosine",
                        "feature_total",
                        "valid_tokens",
                        "windows",
                        "sources",
                    )
                }
            ),
            best={
                "step": selector.best_step,
                "value": selector.best_value,
                "checkpoint_sha256": selection.get("checkpoint_sha256"),
                "metric": "validation_token_total",
            },
        )
        return sha

    selection: dict[str, Any] | None = None
    last_checkpoint_path: Path | None = None
    final_validation_skipped_reason: str | None = None
    try:
        while updates < stop_target:
            if stop_requested["value"]:
                resource_blocked = "sigterm"
                break
            stage_wall = time.perf_counter() - run_started
            if stage_wall > wall_cap:
                resource_blocked = "wall_time_cap"
                break
            if trainer.anchor_presentations + settings.effective_batch > hard_anchors:
                resource_blocked = "anchor_cap"
                break
            used = owned_run_bytes(root)
            if used + reserve > output_cap:
                resource_blocked = "output_cap"
                break
            batch_keys = sampler.next_batch()
            batch = [
                runtime["dataset"].anchor_sample(source_id, endpoint)
                for source_id, endpoint in batch_keys
            ]
            lr_used = float(optimizer.param_groups[0]["lr"])
            result = trainer.run_accumulated_update(
                batch, optimizer=optimizer, scheduler=scheduler, include_action=False
            )
            updates = int(result["updates"])
            lr_next = float(optimizer.param_groups[0]["lr"])
            stage_wall = time.perf_counter() - run_started
            record = {
                "schema": UPDATE_SCHEMA,
                "event": "update",
                "timestamp": _now_iso(),
                "cumulative_updates": updates,
                "new_stage_updates": updates - PARENT_UPDATES,
                "parent_cumulative_updates": PARENT_UPDATES,
                "anchor_presentations": int(result["anchors"]),
                "new_stage_anchor_presentations": int(result["anchors"]) - PARENT_ANCHORS,
                "sampled_source_endpoints": [
                    {"source_id": source_id, "endpoint": int(endpoint)}
                    for source_id, endpoint in batch_keys
                ],
                "feature_mse": float(result["feature_mse"]),
                "feature_cosine": float(result["feature_cosine"]),
                "feature_total": float(result["feature_total"]),
                "loss_total": float(result["feature_total"]),
                "lr_used": lr_used,
                "lr_next": lr_next,
                "gradient_clip": float(settings.gradient_clip),
                "trainable_parameter_count": int(
                    sum(
                        parameter.numel()
                        for parameter in runtime["student"].parameters()
                        if parameter.requires_grad
                    )
                ),
                "trainable_tensor_count": int(
                    sum(
                        1
                        for parameter in runtime["student"].parameters()
                        if parameter.requires_grad
                    )
                ),
                "finite_gradients": all(
                    bool(item.get("finite")) for item in result["gradient_summary"].values()
                ),
                "all_gradients_nonzero": all(
                    bool(item.get("nonzero")) for item in result["gradient_summary"].values()
                ),
                "gradient_summary": dict(result["gradient_summary"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "new_stage_wall_seconds": stage_wall,
                "gpu_allocated_bytes": (
                    int(__import__("torch").cuda.memory_allocated())
                    if __import__("torch").cuda.is_available()
                    else None
                ),
                "gpu_reserved_bytes": (
                    int(__import__("torch").cuda.memory_reserved())
                    if __import__("torch").cuda.is_available()
                    else None
                ),
                "host_rss_bytes": _host_rss_bytes(),
                "output_bytes": used,
                "checkpoint": None,
            }
            _append_jsonl(paths["updates"], record)
            heartbeat.update(cumulative_updates=updates)
            status.update(
                status="running",
                phase="running",
                cumulative_updates=updates,
                anchor_presentations=int(result["anchors"]),
                new_stage_updates=updates - PARENT_UPDATES,
                new_stage_anchor_presentations=int(result["anchors"]) - PARENT_ANCHORS,
                new_stage_wall_seconds=stage_wall,
                output_bytes=used,
                error=None,
            )
            del batch
            if stop_requested["value"]:
                resource_blocked = "sigterm"
                break
            point_due = (
                updates % interval == 0 or updates in MILESTONE_STEPS or updates >= stop_target
            )
            if point_due:
                allow_validation = (time.perf_counter() - run_started) <= wall_cap
                try:
                    _persist_point(
                        updates,
                        final=updates >= stop_target,
                        allow_validation=allow_validation,
                    )
                except _ResourceStop as exc:
                    resource_blocked = str(exc)
                    break
                if stop_requested["value"]:
                    resource_blocked = "sigterm"
                    break
        # Always persist a final cumulative checkpoint, even after a clean stop.
        stage_wall = time.perf_counter() - run_started
        needs_final = (
            last_checkpoint is None
            or int(last_checkpoint["step"]) != updates
            or last_checkpoint.get("kind") != "final"
        )
        if needs_final:
            allow_validation = stage_wall <= wall_cap
            if not allow_validation:
                final_validation_skipped_reason = "wall_time_cap"
            try:
                _persist_point(updates, final=True, allow_validation=allow_validation)
            except _ResourceStop as exc:
                resource_blocked = resource_blocked or str(exc)
        if selection is None:
            selection = _publish_selection()
        final_status = (
            "completed"
            if updates >= cap and resource_blocked is None
            else ("stopped" if resource_blocked == "sigterm" else "stopped_at_update_limit")
        )
        status.update(
            status=final_status,
            phase="finished",
            cumulative_updates=updates,
            new_stage_updates=updates - PARENT_UPDATES,
            anchor_presentations=trainer.anchor_presentations,
            new_stage_anchor_presentations=trainer.anchor_presentations - PARENT_ANCHORS,
            new_stage_wall_seconds=stage_wall,
            output_bytes=owned_run_bytes(root),
            last_checkpoint=last_checkpoint,
            last_validation=(
                None
                if last_validation is None
                else {
                    key: last_validation[key]
                    for key in (
                        "step",
                        "checkpoint_sha256",
                        "feature_mse",
                        "feature_cosine",
                        "feature_total",
                        "valid_tokens",
                        "windows",
                        "sources",
                    )
                }
            ),
            best=(
                None
                if selector.best_step is None
                else {
                    "step": selector.best_step,
                    "value": selector.best_value,
                    "metric": "validation_token_total",
                    "checkpoint_sha256": selection.get("checkpoint_sha256"),
                }
            ),
            stop_reason=resource_blocked,
            final_validation_skipped_reason=final_validation_skipped_reason,
            error=None,
        )
        status.finish(
            status=final_status,
            native_exit=0,
            error=None,
        )
        return {
            "status": final_status,
            "run_root": str(root),
            "updates": updates,
            "new_updates": updates - new_stage_start_updates,
            "cumulative_updates": updates,
            "anchor_presentations": trainer.anchor_presentations,
            "selected_checkpoint_step": selector.best_step,
            "selected_checkpoint_value": selector.best_value,
            "selected_checkpoint_path": selection.get("checkpoint_path"),
            "selected_checkpoint_sha256": selection.get("checkpoint_sha256"),
            "final_checkpoint": last_checkpoint,
            "stop_after_update": stop_target,
            "declared_cumulative_cap": cap,
            "resource_blocked": resource_blocked,
            "early_stopped": earlystop,
            "action_loss_used": False,
            "closed_loop": "not_run",
            "control_restored": False,
            "new_stage_wall_seconds": stage_wall,
            "output_bytes": owned_run_bytes(root),
        }
    except Exception:
        status.finish(status="error", native_exit=2, error=sys.exc_info()[1].__class__.__name__)
        raise
    finally:
        heartbeat.stop()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _prune_checkpoints(
    checkpoints: Path,
    *,
    current_step: int,
    best_step: int,
    final_step: int | None,
) -> None:
    entries = []
    for path in sorted(checkpoints.glob("checkpoint_*.pt")):
        entries.append({"name": path.name})
    plan = plan_checkpoint_retention(
        entries,
        current_step=current_step,
        best_step=best_step,
        final_step=final_step,
    )
    for name in plan["delete"]:
        path = checkpoints / name
        if path.is_file() and not path.is_symlink():
            path.unlink(missing_ok=True)


def _host_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    # Linux reports KiB.
    return value * 1024 if sys.platform.startswith("linux") else value


# --------------------------------------------------------------------------- #
# status / action diagnostic
# --------------------------------------------------------------------------- #
def status(*, run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    value = _status_from_run(root)
    selection_path = root / "selection.json"
    selection = _load_json(selection_path) if selection_path.is_file() else None
    prepared_path = root / "prepared.json"
    prepared = _load_json(prepared_path) if prepared_path.is_file() else None
    return {
        "status": value.get("status"),
        "phase": value.get("phase"),
        "run_root": str(root),
        "cumulative_updates": value.get("cumulative_updates"),
        "new_stage_updates": value.get("new_stage_updates"),
        "target_cumulative_updates": value.get("target_cumulative_updates"),
        "anchor_presentations": value.get("anchor_presentations"),
        "new_stage_wall_seconds": value.get("new_stage_wall_seconds"),
        "new_stage_wall_cap_seconds": value.get("new_stage_wall_cap_seconds"),
        "output_bytes": owned_run_bytes(root),
        "output_cap_bytes": value.get("output_cap_bytes"),
        "last_checkpoint": value.get("last_checkpoint"),
        "last_validation": value.get("last_validation"),
        "best": value.get("best"),
        "native_exit": value.get("native_exit"),
        "error": value.get("error"),
        "selection": selection,
        "prepared": prepared is not None,
        "closed_loop": "not_run",
        "closed_loop_episodes": 0,
        "control_restored": False,
    }


def action_diagnostic(
    *,
    config_path: str | Path,
    run_root: str | Path,
    checkpoint: str | Path,
    step: int | None,
    device: str | None,
) -> dict[str, Any]:
    """Read-only action-MAE diagnostic; never writes selection or stopping state."""
    from owac.evaluation.raw_temporal_distillation import (
        TemporalPureRuntime,
        evaluate_temporal_samples,
    )
    from owac.raw.temporal_packets import RawTemporalDataset
    from owac.training.raw_temporal_distillation import TemporalTrainingConfig

    config = _load_json(config_path)
    root = Path(run_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    payload = _load_parent_payload(checkpoint_path)
    checkpoint_sha = _sha256_file(checkpoint_path)
    cumulative = int(payload.get("cumulative_updates", -1))
    if step is not None and int(step) != cumulative:
        raise ValueError("action diagnostic step differs from the checkpoint")
    runtime = _build_assets(config, device=device)
    settings = TemporalTrainingConfig.from_mapping(config)
    parent_config = _load_json(
        REPO_ROOT / "outputs/raw-distillation-pilot-20261021-101725/authorized-config.json"
    )
    trainer = _trainer(
        runtime,
        parent_config if cumulative == PARENT_UPDATES else config,
        run_identity=dict(payload.get("run_identity") or {}),
    )
    trainer.load_checkpoint(checkpoint_path)
    dataset = RawTemporalDataset(
        runtime["data"], runtime["student"], stats=runtime["bundle"].stats, arm="C"
    )
    samples = []
    for source in runtime["data"].manifest.sources_for_split("validation"):
        samples.extend(dataset.source_samples(source.source_demo_id))
    pure = TemporalPureRuntime(
        student=runtime["student"],
        bridge=runtime["bridge"],
        stats=runtime["bundle"].stats,
        freeze=True,
    )
    record = evaluate_temporal_samples(
        pure,
        samples,
        split="validation",
        step=cumulative,
        noise_seed=int(settings.seed),
    )
    result = {
        "schema": ACTION_DIAGNOSTIC_SCHEMA,
        "created_at": _now_iso(),
        "step": cumulative,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "split": "validation",
        "noise_seed": int(settings.seed),
        "metrics": {
            "first1_mae7": record.step1_mae7,
            "first5_mae7": record.first5_mae7,
            "full10_mae7": record.full10_mae7,
            "groups": dict(record.groups),
            "padding_mae": record.padding_mae,
            "valid_step1": record.valid_step1,
            "valid_first5": record.valid_first5,
            "valid_full10": record.valid_full10,
        },
        "used_for_selection": False,
        "used_for_stop": False,
        "closed_loop": "not_run",
        "closed_loop_episodes": 0,
        "control_restored": False,
        "note": "action MAE diagnostic only; prior pilot action regression is preserved",
    }
    target = root / "action-diagnostic" / f"action-{cumulative:06d}.json"
    _atomic_json(target, result)
    return {
        "status": "diagnostic_written",
        "path": str(target),
        "step": cumulative,
        "checkpoint_sha256": checkpoint_sha,
        "first5_mae7": record.first5_mae7,
        "used_for_selection": False,
        "closed_loop": "not_run",
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--migration", required=True)
    prepare_parser.add_argument("--authority", required=True)
    prepare_parser.add_argument("--parity-receipt", default=None)
    prepare_parser.add_argument("--resource-receipt", default=None)
    prepare_parser.add_argument("--device", default="cuda:0")
    prepare_parser.add_argument("--run-root", default=None)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--migration", required=True)
    train_parser.add_argument("--authority", required=True)
    train_parser.add_argument("--parity-receipt", default=None)
    train_parser.add_argument("--resource-receipt", default=None)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--run-root", default=None)
    train_parser.add_argument("--stop-after-update", type=int, default=None)
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--acceptance", default=None)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--config", default=None)
    status_parser.add_argument("--run-root", required=True)

    diagnostic_parser = sub.add_parser("action-diagnostic")
    diagnostic_parser.add_argument("--config", required=True)
    diagnostic_parser.add_argument("--run-root", required=True)
    diagnostic_parser.add_argument("--checkpoint", required=True)
    diagnostic_parser.add_argument("--step", type=int, default=None)
    diagnostic_parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare(
                config_path=args.config,
                migration_path=args.migration,
                authority_path=args.authority,
                parity_receipt=args.parity_receipt,
                resource_receipt=args.resource_receipt,
                device=args.device,
                run_root=args.run_root,
            )
        elif args.command == "train":
            value = train(
                config_path=args.config,
                migration_path=args.migration,
                authority_path=args.authority,
                parity_receipt=args.parity_receipt,
                resource_receipt=args.resource_receipt,
                stop_after_update=args.stop_after_update,
                resume=args.resume,
                acceptance=args.acceptance,
                device=args.device,
                run_root=args.run_root,
            )
        elif args.command == "status":
            value = status(run_root=args.run_root)
        elif args.command == "action-diagnostic":
            value = action_diagnostic(
                config_path=args.config,
                run_root=args.run_root,
                checkpoint=args.checkpoint,
                step=args.step,
                device=args.device,
            )
        else:  # pragma: no cover - argparse enforces the set
            raise ValueError("unknown command")
    except Exception as exc:
        _fail_prepare_lifecycle(exc)
        print(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "optimizer_updates": None,
                    "optimizer_update_evidence": "unknown; inspect updates.jsonl and checkpoints",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
