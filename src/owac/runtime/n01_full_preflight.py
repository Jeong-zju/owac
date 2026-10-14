"""N01 v1.0 P0 preflight: inventory, profiling driver, source bundle and bundle verify.

The four subcommands deliberately separate read-only inspection from packaging:

``inventory``
    Read the protocol config, the four split manifests and (optionally) the data root;
    emit a JSON inventory plus the train/calibration/validation/test summary.
``profile``
    Drive GPU profiling collection and aggregate raw points. The local CPU path is
    refused with an explicit error instead of producing fabricated GPU numbers.
``bundle``
    Explicitly pack the worktree (``src/owac``, this stage's configs, the needed tests,
    upstream licences/sources and the locks) into ``source.tar.gz`` +
    ``bundle-manifest.json``. Files are walked from the filesystem, so new untracked
    sources are included; ``git archive`` is never used.
``verify``
    Hash and closure checks of a produced bundle. A missing file or a hash mismatch is a
    failure (exit 1), never a warning.

Target form from ``docs/prompts/n01_v1_0/P0_FOUNDATION.md``, implemented here but not
shipped as a ready CLI by the prompts::

    python -m owac.runtime.n01_full_preflight inventory \
        --config configs/encoders/n01_full/protocol_v1.json --output outputs/n01-full/p0-inventory
    python -m owac.runtime.n01_full_preflight profile \
        --config configs/encoders/n01_full/protocol_v1.json --output outputs/n01-full/p0-profile

Only the standard library is imported at module level; ``torch`` and ``numpy`` are
imported lazily inside ``profile collect`` so that ``--help``, ``inventory`` and
``verify`` run in a pure CPU environment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import itertools
import json
import math
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

FORMAT = "owac-n01-full-preflight-v1"
ARCHIVE_NAME = "source.tar.gz"
MANIFEST_NAME = "bundle-manifest.json"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_UNAVAILABLE = 3

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = "configs/encoders/n01_full/protocol_v1.json"
DEFAULT_UPSTREAM_ROOT = "outputs/n01-v1-prompt-preparation/upstream"

REQUIRED_SPLITS = ("train", "calibration", "validation", "test")

FULL_SIZE_DBSR = {
    "channels": 64,
    "num_blocks": 9,
    "feature_width": 512,
    "readout_tokens": 256,
    "readout_width": 2048,
}
PROFILING_BACKBONES = ("dbsr", "bsrt")

BUNDLE_DIRECTORIES = (
    "src/owac",
    "configs/encoders/n01_full",
    "configs/backends/n01-full-runtime",
    "tests/encoders/n01_full",
    "data/manifests/n01-full-v1",
)
BUNDLE_FILES = ("docs/prompts/n01_v1_0/source_manifest.json",)

REQUIRED_BUNDLE_FILES = (
    "src/owac/runtime/n01_full_preflight.py",
    "configs/encoders/n01_full/protocol_v1.json",
    "configs/encoders/n01_full/upstream_lock.json",
    "data/manifests/n01-full-v1/n01-full-v1-index.json",
)

REQUIRED_PROTOCOL_KEYS = (
    "protocol_version",
    "stage",
    "stage_status",
    "input_identity",
    "coordinate_units",
    "data",
    "seeds",
    "unique_source_policy",
    "exact_replay",
    "models_and_upstream",
    "arms",
    "metrics",
    "thresholds",
    "validation_interval",
    "platform_rules",
    "budget",
    "statistics",
    "multi_comparison",
    "backend_protocol",
    "profiling",
)

DENIED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
        "node_modules",
        ".idea",
        ".vscode",
    }
)
DENIED_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".h5",
    ".hdf5",
    ".pem",
    ".key",
)
DENIED_NAME_PREFIXES = (".env", "id_rsa", "id_ed25519", "credentials", "secrets")


class PreflightRefused(RuntimeError):
    """The requested operation is refused with an explicit reason."""


class PreflightUnavailable(RuntimeError):
    """A required environment, module or input is not available here."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 of ``path`` as lowercase hex."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: Any, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def git_output(source_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def normalize_rel(path: str) -> str:
    return PurePosixPath(str(path).replace("\\", "/")).as_posix()


def denied_reason(rel: str) -> str | None:
    parts = PurePosixPath(rel).parts
    for part in parts:
        if part in DENIED_DIR_NAMES:
            return f"directory:{part}"
    name = parts[-1] if parts else ""
    if name.startswith(DENIED_NAME_PREFIXES):
        return "sensitive-name"
    if name.endswith(DENIED_SUFFIXES):
        return "artifact-or-binary"
    return None


def hash_status(declared: Any, computed: str | None) -> str:
    if not declared:
        return "missing-declaration"
    if isinstance(declared, str) and declared.startswith("pending_"):
        return "pending"
    if computed is None:
        return "not-computed"
    return "match" if str(declared) == computed else "mismatch"


def value_counts(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def distinct_count(rows: list[dict], *keys: str) -> int | None:
    if not rows:
        return None
    values = set()
    for row in rows:
        if all(row.get(key) is None for key in keys):
            continue
        values.add(tuple(str(row.get(key)) for key in keys))
    return len(values)


# --------------------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------------------


def manifest_rows(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    for key in ("rows", "entries", "inventory", "files"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return list(value)
    return []


def region_paths_from_payload(payload: Any, region: str) -> set[str]:
    """Relative paths of one region from a manifest's ``sides`` (fallback: ``rows``)."""
    if not isinstance(payload, dict):
        return set()
    sides = payload.get("sides")
    if isinstance(sides, dict) and isinstance(sides.get(region), list):
        return {normalize_rel(str(item)) for item in sides[region] if str(item)}
    paths = set()
    for row in manifest_rows(payload):
        row_region = row.get("split", row.get("side"))
        path = row.get("relative_path", row.get("path", row.get("file")))
        if str(row_region) == region and path:
            paths.add(normalize_rel(str(path)))
    return paths


def count_npz_tree(base: Path, limit: int = 200_000) -> dict:
    base = Path(base)
    report = {"exists": base.is_dir(), "count": 0, "walked_files": 0, "truncated": False}
    if not report["exists"]:
        return report
    count = 0
    walked = 0
    stack: list[Path] = [base]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                stack.append(entry)
                continue
            walked += 1
            if walked > limit:
                report.update({"count": count, "walked_files": walked, "truncated": True})
                return report
            if entry.suffix == ".npz":
                count += 1
    report.update({"count": count, "walked_files": walked})
    return report


def scan_data_root(
    root: Path,
    *,
    raw_roots: dict | None = None,
    mirror_roots: dict | None = None,
    limit: int = 200_000,
) -> dict:
    """Count NPZ files under the declared raw roots, never the RGB mirror twins.

    ``data/<format>/rgb`` mirrors ``data/<format>/raw`` 1:1, so counting the whole
    data root doubles every number.  The mirror is reported separately for the
    1:1 check and never enters a measurement or unique-source count.
    """
    root = Path(root)
    report: dict[str, Any] = {
        "path": str(root),
        "exists": root.is_dir(),
        "raw_counts": {},
        "rgb_mirror_counts": {},
        "walked_files": 0,
        "truncated": False,
        "note": None,
    }
    if not report["exists"]:
        return report
    if raw_roots:
        walked = 0
        truncated = False
        for platform, subdir in sorted(raw_roots.items()):
            scanned = count_npz_tree(root / str(subdir), limit=limit)
            report["raw_counts"][str(platform)] = scanned
            walked += scanned["walked_files"]
            truncated = truncated or scanned["truncated"]
        for platform, subdir in sorted((mirror_roots or {}).items()):
            report["rgb_mirror_counts"][str(platform)] = count_npz_tree(
                root / str(subdir), limit=limit
            )
            walked += report["rgb_mirror_counts"][str(platform)]["walked_files"]
            truncated = truncated or report["rgb_mirror_counts"][str(platform)]["truncated"]
        report["walked_files"] = walked
        report["truncated"] = truncated
        report["note"] = "counts cover the declared raw roots; rgb mirrors reported separately"
        return report
    counted = count_npz_tree(root, limit=limit)
    report["walked_files"] = counted["walked_files"]
    report["truncated"] = counted["truncated"]
    report["note"] = "no raw_roots declared; the whole root was counted"
    report["raw_counts"] = {"<root>": counted}
    return report


def declared_platform_expected(data_section: dict, platform: str) -> Any:
    platforms = data_section.get("platforms")
    spec = platforms.get(platform) if isinstance(platforms, dict) else None
    if not isinstance(spec, dict):
        return None
    return spec.get("expected_npz_frozen", spec.get("expected_npz"))


def inventory(
    config_path: Path,
    output: Path,
    *,
    repo_root: Path,
    data_root: Path | None,
    check_loader: bool = False,
) -> dict:
    config_path = Path(config_path)
    repo_root = Path(repo_root)
    resolved_config = config_path if config_path.is_absolute() else repo_root / config_path
    if not resolved_config.is_file():
        raise PreflightUnavailable(f"protocol config not found: {resolved_config}")
    protocol = read_json(resolved_config)
    data_section = protocol.get("data") if isinstance(protocol.get("data"), dict) else {}
    splits_declared = (
        data_section.get("splits") if isinstance(data_section.get("splits"), dict) else {}
    )
    checks: list[dict] = []

    def check(check_id: str, ok: bool | None, detail: str = "") -> None:
        checks.append({"id": check_id, "ok": ok, "detail": detail})

    # --- platform manifests ---------------------------------------------------------
    platform_specs = data_section.get("platform_manifests")
    if not isinstance(platform_specs, dict) or not platform_specs:
        platform_specs = {}
        check(
            "protocol:platform_manifests_declared",
            False,
            "data.platform_manifests is missing; the four regions live inside the "
            "per-platform manifests",
        )
    else:
        check("protocol:platform_manifests_declared", True, ", ".join(sorted(platform_specs)))

    platform_reports: dict[str, dict] = {}
    loaded: dict[str, dict] = {}
    for platform, spec in sorted(platform_specs.items()):
        spec = spec if isinstance(spec, dict) else {}
        rel = spec.get("manifest_path")
        declared = spec.get("manifest_sha256")
        report: dict[str, Any] = {
            "platform": platform,
            "manifest_path": rel,
            "declared_sha256": declared,
            "declared_hash_status": None,
            "exists": False,
            "sha256": None,
            "region_counts": {},
            "row_count": 0,
            "total_files": 0,
            "status": "missing",
        }
        if rel:
            path = Path(rel)
            path = path if path.is_absolute() else repo_root / path
            if path.is_file():
                payload = read_json(path)
                regions = {
                    region: region_paths_from_payload(payload, region) for region in REQUIRED_SPLITS
                }
                rows = manifest_rows(payload)
                report.update(
                    {
                        "exists": True,
                        "sha256": sha256_file(path),
                        "declared_hash_status": hash_status(declared, sha256_file(path)),
                        "region_counts": {region: len(paths) for region, paths in regions.items()},
                        "row_count": len(rows),
                        "total_files": sum(len(paths) for paths in regions.values()),
                        "status": "ok" if all(regions.values()) else "incomplete",
                    }
                )
                loaded[platform] = {"regions": regions, "rows": rows, "payload": payload}
        # Bind every auxiliary evidence file, not just the top-level manifest.
        for kind in ("inventory", "loader_evidence"):
            if spec.get(f"{kind}_path"):
                auxiliary = repo_root / spec[f"{kind}_path"]
                check(
                    f"manifest:{platform}:{kind}_sha256",
                    auxiliary.is_file()
                    and hash_status(spec.get(f"{kind}_sha256"), sha256_file(auxiliary)) == "match",
                    str(auxiliary),
                )
        platform_reports[platform] = report
        check(f"manifest:{platform}:present", report["exists"], str(rel))
        if not report["exists"]:
            continue
        status = report["declared_hash_status"]
        if status in ("pending", "missing-declaration"):
            check(f"manifest:{platform}:sha256", None, str(status))
        else:
            check(f"manifest:{platform}:sha256", status == "match", str(status))
        for region in REQUIRED_SPLITS:
            declared_region = splits_declared.get(region)
            declared_counts = (
                declared_region.get("counts") if isinstance(declared_region, dict) else None
            )
            declared_count = (
                declared_counts.get(platform) if isinstance(declared_counts, dict) else None
            )
            computed = report["region_counts"].get(region, 0)
            if declared_count is None or (
                isinstance(declared_count, str) and declared_count.startswith("pending_")
            ):
                check(
                    f"count:{platform}:{region}",
                    None,
                    f"computed={computed}, declared={declared_count}",
                )
            else:
                check(
                    f"count:{platform}:{region}",
                    computed == declared_count,
                    f"computed={computed}, declared={declared_count}",
                )

    # --- region structure per platform ----------------------------------------------
    for platform, report in sorted(platform_reports.items()):
        for region in REQUIRED_SPLITS:
            if not report["exists"]:
                check(f"region:{platform}:{region}:nonempty", None, "manifest missing")
                continue
            paths = loaded[platform]["regions"].get(region, set())
            check(f"region:{platform}:{region}:nonempty", bool(paths), f"{len(paths)} paths")
        if not report["exists"]:
            check(f"regions:{platform}:disjoint", None, "manifest missing")
            check(f"regions:{platform}:union", None, "manifest missing")
            continue
        regions = loaded[platform]["regions"]
        pairs = list(itertools.combinations(REQUIRED_SPLITS, 2))
        overlaps = {
            f"{left}|{right}": len(regions.get(left, set()) & regions.get(right, set()))
            for left, right in pairs
        }
        empty = [region for region in REQUIRED_SPLITS if not regions.get(region)]
        if empty:
            check(
                f"regions:{platform}:disjoint",
                False,
                f"empty region(s) make disjointness vacuous: {empty}",
            )
        else:
            check(
                f"regions:{platform}:disjoint",
                sum(overlaps.values()) == 0,
                f"overlaps={overlaps}",
            )
        summed = sum(len(regions.get(region, set())) for region in REQUIRED_SPLITS)
        union: set[str] = set()
        for region in REQUIRED_SPLITS:
            union |= regions.get(region, set())
        expected = declared_platform_expected(data_section, platform)
        rows = loaded[platform]["rows"]
        row_paths = {
            normalize_rel(str(row.get("relative_path", row.get("path"))))
            for row in rows
            if row.get("relative_path") or row.get("path")
        }
        parts = [f"sum={summed}", f"union={len(union)}", f"expected={expected}"]
        ok = summed == len(union)
        if isinstance(expected, int):
            ok = ok and len(union) == expected and summed == expected
        if row_paths:
            ok = ok and row_paths == union
            parts.append(f"rows={len(row_paths)}")
        check(f"regions:{platform}:union", ok, "; ".join(parts))
        group_sides: dict[str, set[str]] = {}
        row_sides = {region: [] for region in REQUIRED_SPLITS}
        for row in rows:
            group_sides.setdefault(str(row.get("source_group")), set()).add(row.get("split"))
            if row.get("split") in row_sides:
                row_sides[row["split"]].append(row["relative_path"])
        check(
            f"regions:{platform}:source_groups_disjoint",
            bool(rows)
            and all(len(tags) == 1 for tags in group_sides.values())
            and all(row.get("source_group") for row in rows),
            "source identities may not cross partition boundaries",
        )
        check(
            f"regions:{platform}:rows_match_sides",
            len(rows) == len(row_paths)
            and set(loaded[platform]["payload"].get("sides", {})) == set(REQUIRED_SPLITS)
            and all(
                sorted(row_sides[r]) == sorted(loaded[platform]["payload"]["sides"][r])
                for r in REQUIRED_SPLITS
            ),
            "exact row membership and duplicate-free sides",
        )

    # --- cross-platform region report ------------------------------------------------
    regions_report: dict[str, dict] = {}
    for region in REQUIRED_SPLITS:
        per_platform: dict[str, dict] = {}
        merged_rows: list[dict] = []
        for platform, bundle in loaded.items():
            paths = bundle["regions"].get(region, set())
            rows = [
                row
                for row in bundle["rows"]
                if str(row.get("split", row.get("side", region))) == region
            ]
            merged_rows.extend(rows)
            per_platform[platform] = {"file_count": len(paths), "row_count": len(rows)}
        declared_region = (
            splits_declared.get(region) if isinstance(splits_declared.get(region), dict) else {}
        )
        declared_counts = (
            declared_region.get("counts") if isinstance(declared_region.get("counts"), dict) else {}
        )
        total = sum(item["file_count"] for item in per_platform.values())
        declared_total = declared_region.get("cross_platform_total")
        regions_report[region] = {
            "region": region,
            "platforms": per_platform,
            "file_count": total,
            "cross_platform_total": total,
            "declared_counts": declared_counts,
            "declared_cross_platform_total": declared_total,
            "datasets": value_counts(merged_rows, "dataset_id")
            or value_counts(merged_rows, "dataset"),
            "storage_encodings": value_counts(merged_rows, "input_mode"),
            "source_group_count": distinct_count(merged_rows, "source_group"),
            "episode_count": distinct_count(merged_rows, "episode_id")
            or distinct_count(merged_rows, "source_group", "episode"),
            "camera_count": distinct_count(merged_rows, "camera_id"),
            "units": value_counts(merged_rows, "units"),
            "calibration_versions": value_counts(merged_rows, "calibration_version"),
            "status": "ok" if total > 0 else "empty_or_missing",
        }
        if declared_total is None or (
            isinstance(declared_total, str) and declared_total.startswith("pending_")
        ):
            check(
                f"split:{region}:cross_platform_total",
                None,
                f"computed={total}, declared={declared_total}",
            )
        else:
            check(
                f"split:{region}:cross_platform_total",
                total == declared_total,
                f"computed={total}, declared={declared_total}",
            )

    # --- manifest index --------------------------------------------------------------
    index_spec = (
        data_section.get("manifest_index")
        if isinstance(data_section.get("manifest_index"), dict)
        else {}
    )
    index_report: dict[str, Any] = {
        "path": index_spec.get("manifest_path"),
        "exists": False,
        "sha256": None,
    }
    if index_spec:
        rel = index_spec.get("manifest_path")
        declared = index_spec.get("sha256")
        if rel:
            path = Path(rel)
            path = path if path.is_absolute() else repo_root / path
            if path.is_file():
                index_report.update(
                    {
                        "exists": True,
                        "sha256": sha256_file(path),
                        "declared_sha256": declared,
                        "declared_hash_status": hash_status(declared, sha256_file(path)),
                    }
                )
        check("manifest_index:present", bool(index_report["exists"]), str(rel))
        if index_report["exists"]:
            status = index_report["declared_hash_status"]
            check(
                "manifest_index:sha256",
                None if status in ("pending", "missing-declaration") else status == "match",
                str(status),
            )

    # --- data root (raw roots only) ---------------------------------------------------
    declared_root = data_root if data_root is not None else data_section.get("data_root")
    raw_roots = (
        data_section.get("raw_roots") if isinstance(data_section.get("raw_roots"), dict) else None
    )
    mirror_roots = (
        data_section.get("rgb_mirror_roots")
        if isinstance(data_section.get("rgb_mirror_roots"), dict)
        else None
    )
    root_report = (
        scan_data_root(Path(declared_root), raw_roots=raw_roots, mirror_roots=mirror_roots)
        if declared_root
        else {"path": None, "exists": False, "raw_counts": {}, "rgb_mirror_counts": {}}
    )
    if not root_report.get("exists"):
        check(
            "data_root:present",
            None,
            f"{root_report.get('path')} not reachable from this host; raw counts stay pending",
        )
    else:
        check("data_root:present", True, str(root_report.get("path")))
        check("data_root:complete_scan", not root_report.get("truncated"), "scan limit")
        for platform, payload in loaded.items():
            base = Path(declared_root) / (raw_roots or {}).get(platform, "")
            observed_paths = {
                p.relative_to(base).as_posix() for p in base.rglob("*.npz") if p.is_file()
            }
            expected_paths = {row["relative_path"] for row in payload["rows"]}
            check(
                f"data_root:{platform}:exact_paths",
                observed_paths == expected_paths,
                f"missing={len(expected_paths - observed_paths)}, "
                f"extra={len(observed_paths - expected_paths)}",
            )
            drift = []
            missing_fingerprints = []
            for row in payload["rows"]:
                if not {"size_bytes", "mtime_ns", "inventory_hash"} <= row.keys():
                    missing_fingerprints.append(row["relative_path"])
                    continue
                path = base / row["relative_path"]
                if not path.is_file():
                    drift.append(row["relative_path"])
                    continue
                stat = path.stat()
                if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
                    drift.append(row["relative_path"])
            check(
                f"data_root:{platform}:inventory_fingerprint",
                False if drift else (None if missing_fingerprints else True),
                f"stat drift={len(drift)}, undeclared={len(missing_fingerprints)}; "
                "not a content hash",
            )
            mirror_path = (mirror_roots or {}).get(platform)
            if mirror_path:
                mirror_base = Path(declared_root) / mirror_path
                mirror_paths = {
                    p.relative_to(mirror_base).as_posix()
                    for p in mirror_base.rglob("*.npz")
                    if p.is_file()
                }
                check(
                    f"data_root:{platform}:rgb_mirror_paths",
                    mirror_paths == observed_paths,
                    "relative path sets must match, not only counts",
                )
        for platform, scanned in sorted(root_report.get("raw_counts", {}).items()):
            observed = scanned.get("count")
            expected = declared_platform_expected(data_section, platform)
            if isinstance(expected, int):
                check(
                    f"data_root:{platform}:raw_npz_count",
                    observed == expected,
                    f"observed={observed}, expected={expected}",
                )
            else:
                check(
                    f"data_root:{platform}:raw_npz_count",
                    None,
                    f"observed={observed}, expected={expected}",
                )
            mirror = root_report.get("rgb_mirror_counts", {}).get(platform)
            if mirror is not None:
                check(
                    f"data_root:{platform}:rgb_mirror_1to1",
                    observed == mirror.get("count"),
                    f"raw={observed}, rgb_mirror={mirror.get('count')}",
                )

    task_spec = protocol.get("statistics", {}).get("task_groups")
    if task_spec:
        task_path = repo_root / task_spec["path"]
        task_ok = task_path.is_file() and sha256_file(task_path) == task_spec["sha256"]
        check("statistics:task_groups_hash", task_ok, str(task_path))
        if task_ok:
            task_map = read_json(task_path)["platforms"]
            for platform, payload in loaded.items():
                task_rows = task_map.get(platform, {}).get("rows", [])
                check(
                    f"statistics:{platform}:task_groups_coverage",
                    len(task_rows) == len(payload["rows"])
                    and {r["relative_path"] for r in task_rows}
                    == {r["relative_path"] for r in payload["rows"]}
                    and all(r.get("task_id") for r in task_rows),
                    "one task cluster per frozen source path",
                )

    loader_result = None
    if check_loader:
        from owac.runtime.n01_full_profile import probe_loaders

        try:
            loader_protocol = json.loads(json.dumps(protocol))
            if data_root is not None:
                loader_protocol["data"]["data_root"] = str(data_root)
            loader_result = probe_loaders(loader_protocol, repo_root)
            check(
                "loader:four_regions",
                all(
                    row["paths_equal"] and row["action_finite"]
                    for regions in loader_result.values()
                    for row in regions.values()
                ),
                "real batches using frozen manifest and protocol statistics",
            )
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            loader_result = {"error": str(exc)}
            check("loader:four_regions", False, str(exc))

    failures = [item for item in checks if item["ok"] is False]
    pending = [item for item in checks if item["ok"] is None]
    region_totals = {region: regions_report[region]["file_count"] for region in REQUIRED_SPLITS}
    summary = {
        "status": "failed" if failures else ("pending" if pending else "ok"),
        "platforms_present": [
            platform for platform, report in platform_reports.items() if report["exists"]
        ],
        "missing_manifests": [
            platform for platform, report in platform_reports.items() if not report["exists"]
        ],
        "region_totals": region_totals,
        "cross_platform_total": sum(region_totals.values()),
        "failures": [item["id"] for item in failures],
        "pending": [item["id"] for item in pending],
    }
    result = {
        "format": FORMAT,
        "kind": "inventory",
        "loader_probe": loader_result,
        "generated_at": now_iso(),
        "config": {
            "path": str(resolved_config),
            "sha256": sha256_file(resolved_config),
            "protocol_version": protocol.get("protocol_version"),
            "stage": protocol.get("stage"),
        },
        "repo_root": str(repo_root),
        "data_root": root_report,
        "manifest_index": index_report,
        "platform_manifests": platform_reports,
        "regions": regions_report,
        "splits": regions_report,
        "checks": checks,
        "summary": summary,
    }
    write_json(Path(output) / "inventory.json", result)
    return result


# --------------------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------------------


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_profiling_backbone(protocol: dict, override: str | None = None) -> dict:
    """Refuse a non-full-size DBSR profile; an explicit backbone override is required.

    Mirrors ``ContentConfig.validate`` from ``src/owac/frameworks/n01_full/config.py``:
    the first version profiles the full-size DBSR only.  A smaller/other backbone must
    be requested by changing the backbone field, never by relaxing these sizes.
    """
    profiling = protocol.get("profiling", {}) if isinstance(protocol.get("profiling"), dict) else {}
    backbone = str(override or profiling.get("backbone") or "dbsr")
    if backbone not in PROFILING_BACKBONES:
        raise PreflightRefused(
            f"unknown profiling backbone {backbone!r}; expected one of {PROFILING_BACKBONES}"
        )
    contract = profiling.get("backbone_contract")
    contract = contract if isinstance(contract, dict) else {}
    if backbone == "dbsr":
        wrong = {
            name: contract.get(name, expected)
            for name, expected in FULL_SIZE_DBSR.items()
            if contract.get(name, expected) != expected
        }
        if wrong:
            raise PreflightRefused(
                "non-full-size DBSR profiling is refused: the first version profiles the full "
                f"64/9/512 DBSR with a 256x2048 readout, got {wrong}. Declare a different "
                "profiling.backbone (the P6 BSRT backbone) or fix the sizes; do not silently "
                "relax the full-size contract."
            )
    return {
        "backbone": backbone,
        "contract": {
            name: contract.get(name, expected) for name, expected in FULL_SIZE_DBSR.items()
        },
    }


def load_torch():
    try:
        import torch
    except ImportError as exc:
        raise PreflightUnavailable(
            "profile collect requires the encoder extra (torch==2.9.1 CPU locally); "
            f"import failed: {exc}"
        ) from exc
    return torch


def resolve_factory(spec: str):
    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise PreflightRefused(f"encoder factory spec must be 'module:attr', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise PreflightUnavailable(
            f"encoder factory module unavailable: {module_name}: {exc}"
        ) from exc
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise PreflightUnavailable(f"encoder factory {spec!r} is not callable")
    return factory


def scalar_from_output(output: Any, torch_module: Any):
    if torch_module.is_tensor(output):
        return output.float().square().mean()
    if isinstance(output, dict):
        tensors = [value for value in output.values() if torch_module.is_tensor(value)]
        if tensors:
            return sum(tensor.float().square().mean() for tensor in tensors)
    if isinstance(output, (list, tuple)):
        tensors = [value for value in output if torch_module.is_tensor(value)]
        if tensors:
            return sum(tensor.float().square().mean() for tensor in tensors)
    raise PreflightRefused(
        "encoder output exposes no tensor to differentiate; profile collect needs a "
        "tensor or a dict/list of tensors"
    )


def run_point_step(encoder: Any, tensor: Any, torch_module: Any) -> None:
    output = encoder(tensor) if callable(encoder) else encoder.forward(tensor)
    scalar_from_output(output, torch_module).backward()
    tensor.grad = None
    zero_grad = getattr(encoder, "zero_grad", None)
    if callable(zero_grad):
        zero_grad(set_to_none=True)


def probe_io(io_root: Path, raw_dir: Path, sample: int, torch_module: Any) -> dict | None:
    root = Path(io_root)
    if not root.exists():
        return {"path": str(root), "exists": False, "note": "io root absent; I/O not measured"}
    try:
        import numpy as np
    except ImportError as exc:
        raise PreflightUnavailable(f"I/O probe requires numpy: {exc}") from exc
    files = sorted(path for path in root.rglob("*.npz") if path.is_file())[: max(int(sample), 0)]
    report: dict[str, Any] = {
        "path": str(root),
        "exists": True,
        "files": len(files),
        "bytes": 0,
        "seconds": None,
        "mib_per_second": None,
        "note": None,
    }
    if not files:
        report["note"] = "no *.npz under io root; I/O not measured"
        return report
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    started = time.perf_counter()
    total = 0
    for path in files:
        with np.load(path) as data:
            total += sum(int(data[key].nbytes) for key in data.files)
    elapsed = time.perf_counter() - started
    report["bytes"] = int(total)
    report["seconds"] = elapsed
    report["mib_per_second"] = (total / (1024 * 1024) / elapsed) if elapsed > 0 else None
    write_json(Path(raw_dir) / "io.json", {"kind": "io", **report})
    return report


def profile_collect(
    protocol: dict,
    output: Path,
    *,
    raw_dir: Path,
    factory_spec: str | None = None,
    iterations: int | None = None,
    warmup: int | None = None,
    io_root: Path | None = None,
    io_sample: int = 8,
    backbone_override: str | None = None,
) -> dict:
    validate_profiling_backbone(protocol, backbone_override)
    profiling = protocol.get("profiling", {}) if isinstance(protocol.get("profiling"), dict) else {}
    spec = factory_spec or profiling.get("encoder_factory")
    if not spec:
        raise PreflightRefused(
            "no profiling.encoder_factory in the protocol and no --encoder-factory"
        )
    if str(spec).startswith("pending_"):
        raise PreflightUnavailable(
            f"profiling.encoder_factory is {spec}: P0 does not own "
            "src/owac/frameworks/n01_full/encoder.py (the file-responsibility table excludes "
            "it) and the external N01FullEncoder adapter belongs to P4. Run collect after P4 "
            "lands, or pass --encoder-factory module:attr explicitly."
        )
    torch = load_torch()
    if not torch.cuda.is_available():
        raise PreflightRefused(
            "GPU profiling refused: torch.cuda.is_available() is False on this host. "
            "The local CPU fallback measures nothing and must not be reported as a GPU "
            "profile; run 'profile collect' inside the independent remote CUDA environment "
            "(configs/backends/n01-full-runtime/)."
        )
    factory = resolve_factory(spec)
    device = torch.device("cuda")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0)
    measurement_channels = int(profiling.get("measurement_channels", 4))
    records = []
    for point in profiling.get("points", []):
        point_id = str(point.get("id"))
        resolution = int(point["resolution"])
        history = int(point["history"])
        batch = int(point.get("batch", 1))
        channels = int(point.get("channels", measurement_channels))
        encoder_input_channels = point.get("input_channels")
        if (
            encoder_input_channels is not None
            and int(encoder_input_channels) % measurement_channels
        ):
            raise PreflightRefused(
                f"profiling point {point_id!r} declares input_channels={encoder_input_channels}, "
                f"not a whole multiple of the {measurement_channels} measurement channels"
            )
        point_backbone = validate_profiling_backbone(
            protocol, point.get("backbone") or backbone_override
        )
        warm = int(warmup if warmup is not None else point.get("warmup", 3))
        count = int(iterations if iterations is not None else point.get("iterations", 20))
        torch.manual_seed(int(profiling.get("seed", 101)))
        torch.cuda.empty_cache()
        encoder = factory(
            protocol=protocol,
            point=point,
            device=str(device),
            backbone=point_backbone["backbone"],
        )
        shape = (batch, history, channels, resolution, resolution)
        tensor = torch.rand(*shape, device=device, dtype=torch.float32)
        # RAW is an input, not a trained parameter. Do not allocate its gradient.

        if count < 20 or warm < 1:
            raise PreflightRefused("P0 profiling requires warmup >= 1 and iterations >= 20")
        for _ in range(warm):
            run_point_step(encoder, tensor, torch)
        # Check real parameter gradients outside the timed loop.
        scalar_from_output(encoder(tensor), torch).backward()
        gradients = [p.grad for p in encoder.parameters() if p.requires_grad]
        finite = all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        nonzero = any(g is not None and bool(g.abs().sum() > 0) for g in gradients)
        zero_gradients = [
            name
            for name, p in encoder.named_parameters()
            if p.grad is not None and not bool(p.grad.abs().sum() > 0)
        ]
        if not gradients or not finite or not nonzero:
            raise PreflightRefused("profile gradients must all be finite with nonzero total norm")
        encoder.zero_grad(set_to_none=True)
        del gradients
        forward_timings = []
        with torch.no_grad():
            for _ in range(count):
                torch.cuda.synchronize()
                started = time.perf_counter()
                forward_output = encoder(tensor)
                torch.cuda.synchronize()
                forward_timings.append((time.perf_counter() - started) * 1e3)
                del forward_output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(max(count, 1)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            run_point_step(encoder, tensor, torch)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1e3)
        record = {
            "kind": "profile-point",
            "point_id": point_id,
            "backbone": point_backbone["backbone"],
            "resolution": resolution,
            "history": history,
            "batch": batch,
            "channels": channels,
            "iterations": max(count, 1),
            "warmup": max(warm, 0),
            "device_name": device_name,
            "encoder_factory": spec,
            "entry": getattr(encoder, "entry", "external_factory"),
            "camera_count": 1,
            "gradients": {
                "all_finite": finite,
                "any_nonzero": nonzero,
                "zero_gradient_parameters": zero_gradients,
            },
            "forward_ms": {
                "p50": percentile(forward_timings, 0.5),
                "p95": percentile(forward_timings, 0.95),
            },
            "parameters": sum(p.numel() for p in encoder.parameters()),
            "torch": torch.__version__,
            "input_time_order": "oldest_first",
            "scope": "four-channel CFA backbone/fusion only; no P4 support/replay or policy",
            "timings_ms": {
                "p50": percentile(timings, 0.5),
                "p95": percentile(timings, 0.95),
                "min": min(timings),
                "max": max(timings),
                "mean": sum(timings) / len(timings),
            },
            "memory_bytes": {
                "allocated": int(torch.cuda.memory_allocated()),
                "reserved": int(torch.cuda.memory_reserved()),
                "peak_allocated": int(torch.cuda.max_memory_allocated()),
                "peak_reserved": int(torch.cuda.max_memory_reserved()),
            },
            "input_elements": int(math.prod(shape)),
            "output_bytes": None,
            "timestamp": now_iso(),
        }
        write_json(raw_dir / f"{point_id}.json", record)
        records.append(record)
        del encoder, tensor
        torch.cuda.empty_cache()
    io_report = probe_io(io_root, raw_dir, io_sample, torch) if io_root is not None else None
    summary = profile_summarize(protocol, raw_dir, Path(output), io_report=io_report)
    return {"records": records, "summary": summary}


def profile_summarize(
    protocol: dict, raw_dir: Path, output: Path, *, io_report: dict | None = None
) -> dict:
    raw_dir = Path(raw_dir)
    records = []
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.json")):
            payload = read_json(path)
            if isinstance(payload, dict) and payload.get("kind") == "profile-point":
                records.append(payload)
            elif isinstance(payload, dict) and payload.get("kind") == "io" and io_report is None:
                io_report = payload
    if not records:
        raise PreflightUnavailable(
            f"no raw profiling points under {raw_dir}; run 'profile collect' on a CUDA host "
            "first (the local CPU path never fabricates GPU numbers)"
        )
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["point_id"]), []).append(record)
    points = []
    for point_id, group in sorted(grouped.items()):
        example = group[0]
        p50s = [float(item["timings_ms"]["p50"]) for item in group]
        p95s = [float(item["timings_ms"]["p95"]) for item in group]
        peaks = [int(item["memory_bytes"]["peak_allocated"]) for item in group]
        points.append(
            {
                "point_id": point_id,
                "backbone": example.get("backbone", "dbsr"),
                "resolution": example["resolution"],
                "history": example["history"],
                "batch": example["batch"],
                "channels": example["channels"],
                "iterations": example["iterations"],
                "device_name": example["device_name"],
                "samples": len(group),
                "timings_ms": {
                    "p50": percentile(p50s, 0.5),
                    "p95": percentile(p95s, 0.95),
                    "min": min(p50s),
                    "max": max(p50s),
                },
                "memory_bytes": {"peak_allocated": max(peaks)},
                "step_seconds": (percentile(p50s, 0.5) or 0.0) / 1e3,
            }
        )
    cost_model = protocol.get("profiling", {}).get("cost_model", {})
    units = cost_model.get("units", []) if isinstance(cost_model, dict) else []
    basis_id = cost_model.get("basis_point_id")
    basis = next((p for p in points if p["point_id"] == basis_id), None)
    if basis is None:
        raise PreflightUnavailable(f"cost basis point {basis_id!r} was not measured")
    estimates = []
    for unit in units:
        unit_id = str(unit.get("id"))
        updates = unit.get("updates")
        runs = unit.get("runs")
        note = None
        gpu_hours = None
        if isinstance(updates, (int, float)) and isinstance(runs, (int, float)):
            cameras = unit.get("camera_count")
            accumulation = unit.get("gradient_accumulation")
            if not isinstance(cameras, (int, float)) or not isinstance(accumulation, (int, float)):
                raise PreflightRefused(
                    "cost estimates require explicit camera_count and gradient_accumulation"
                )
            gpu_hours = (
                float(basis["step_seconds"])
                * float(updates)
                * float(runs)
                * cameras
                * accumulation
                / 3600.0
            )
        else:
            note = "updates or runs still pending; estimate deferred"
        estimates.append(
            {
                "unit_id": unit_id,
                "updates": updates,
                "max_updates": unit.get("max_updates"),
                "runs": runs,
                "gpu_hours": gpu_hours,
                "basis_point_id": basis_id,
                "camera_count": unit.get("camera_count"),
                "gradient_accumulation": unit.get("gradient_accumulation"),
                "note": note,
            }
        )
    result = {
        "format": FORMAT,
        "kind": "profile-summary",
        "generated_at": now_iso(),
        "raw_dir": str(raw_dir),
        "basis": (
            "forward+backward p50 of the explicitly selected profiling point; "
            "motion, support, replay, I/O and eval costs are excluded unless measured"
        ),
        "points": points,
        "io": io_report,
        "gpu_hours_estimate": {
            "status": "estimate",
            "estimates": estimates,
            "note": (
                "backbone-only estimate; multiply measured microsteps by cameras, "
                "accumulation and runs"
            ),
        },
        "status": "collected",
    }
    write_json(Path(output) / "profile-summary.json", result)
    return result


def run_profile(
    config_path: Path,
    output: Path,
    *,
    repo_root: Path,
    mode: str,
    raw_dir: Path | None,
    factory_spec: str | None,
    iterations: int | None,
    warmup: int | None,
    io_root: Path | None,
    io_sample: int,
    backbone_override: str | None = None,
) -> dict:
    resolved_config = Path(config_path)
    resolved_config = (
        resolved_config if resolved_config.is_absolute() else repo_root / resolved_config
    )
    if not resolved_config.is_file():
        raise PreflightUnavailable(f"protocol config not found: {resolved_config}")
    protocol = read_json(resolved_config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = Path(raw_dir) if raw_dir is not None else output / "raw"
    if mode == "summarize":
        return profile_summarize(protocol, raw, output)
    if mode == "collect":
        result = profile_collect(
            protocol,
            output,
            raw_dir=raw,
            factory_spec=factory_spec,
            iterations=iterations,
            warmup=warmup,
            io_root=io_root,
            io_sample=io_sample,
            backbone_override=backbone_override,
        )
        return result["summary"]
    has_raw = raw.is_dir() and any(raw.glob("*.json"))
    if has_raw:
        return profile_summarize(protocol, raw, output)
    result = profile_collect(
        protocol,
        output,
        raw_dir=raw,
        factory_spec=factory_spec,
        iterations=iterations,
        warmup=warmup,
        io_root=io_root,
        io_sample=io_sample,
        backbone_override=backbone_override,
    )
    return result["summary"]


# --------------------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------------------


def walk_relative(root: Path, rel: str) -> list[str]:
    base = Path(root) / rel
    if base.is_file():
        return [normalize_rel(rel)]
    if not base.is_dir():
        return []
    return sorted(
        normalize_rel(str(path.relative_to(root))) for path in base.rglob("*") if path.is_file()
    )


def git_metadata(source_root: Path) -> dict:
    head = git_output(source_root, "rev-parse", "HEAD")
    if head is None:
        return {"available": False, "note": "not a git checkout or git unavailable"}
    branch = git_output(source_root, "rev-parse", "--abbrev-ref", "HEAD")
    untracked_raw = (
        git_output(source_root, "ls-files", "--others", "--exclude-standard", "-z") or ""
    )
    modified_raw = git_output(source_root, "ls-files", "--modified", "-z") or ""
    status = git_output(source_root, "status", "--porcelain=v1", "-uall") or ""
    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=source_root,
        capture_output=True,
        check=False,
    )
    diff_digest = hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
    return {
        "available": True,
        "head": head.strip(),
        "branch": (branch or "").strip() or None,
        "status_lines": [line for line in status.splitlines() if line],
        "diff_head_sha256": diff_digest,
        "untracked_files": [normalize_rel(item) for item in untracked_raw.split("\0") if item],
        "modified_files": [normalize_rel(item) for item in modified_raw.split("\0") if item],
    }


def bundle(
    source_root: Path,
    output: Path,
    *,
    include_upstream: bool = True,
    upstream_root: str = DEFAULT_UPSTREAM_ROOT,
) -> dict:
    source_root = Path(source_root).resolve()
    output = Path(output)
    if not source_root.is_dir():
        raise PreflightUnavailable(f"source root not found: {source_root}")
    if output.exists() and any(output.iterdir()):
        raise PreflightRefused(f"bundle output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    missing_includes = []
    candidates: dict[str, str] = {}
    for rel_dir in BUNDLE_DIRECTORIES:
        found = walk_relative(source_root, rel_dir)
        if not found:
            missing_includes.append(rel_dir)
        for rel in found:
            candidates[rel] = "stage-source"
    for rel_file in BUNDLE_FILES:
        if (source_root / rel_file).is_file():
            candidates[normalize_rel(rel_file)] = "upstream-provenance"
        else:
            missing_includes.append(rel_file)
    upstream_included = False
    upstream_files = 0
    if include_upstream:
        upstream_files_list = walk_relative(source_root, upstream_root)
        if upstream_files_list:
            upstream_included = True
            upstream_files = len(upstream_files_list)
            for rel in upstream_files_list:
                candidates[rel] = "upstream-source-or-licence"
    if missing_includes:
        raise PreflightUnavailable(
            "required bundle includes are missing: " + ", ".join(sorted(set(missing_includes)))
        )

    excluded: dict[str, int] = {}
    excluded_examples: list[dict] = []
    files: dict[str, dict] = {}
    for rel, reason in sorted(candidates.items()):
        deny = denied_reason(rel)
        if deny:
            excluded[deny] = excluded.get(deny, 0) + 1
            if len(excluded_examples) < 10:
                excluded_examples.append({"path": rel, "reason": deny})
            continue
        path = source_root / rel
        files[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "selection": reason,
        }

    metadata = git_metadata(source_root)
    untracked_report = {
        "available": bool(metadata.get("available")),
        "included": [],
        "denied": [],
        "missing": [],
    }
    if metadata.get("available"):
        for rel in metadata["untracked_files"]:
            in_scope = any(
                rel == prefix or rel.startswith(prefix.rstrip("/") + "/")
                for prefix in (*BUNDLE_DIRECTORIES, *BUNDLE_FILES)
            )
            if not in_scope:
                continue
            if rel in files:
                untracked_report["included"].append(rel)
            elif denied_reason(rel) is not None:
                untracked_report["denied"].append(rel)
            else:
                untracked_report["missing"].append(rel)
    manifest = {
        "format": FORMAT,
        "kind": "source-bundle",
        "created_at": now_iso(),
        "source_root": str(source_root),
        "archive": ARCHIVE_NAME,
        "git": metadata,
        "include": {
            "directories": list(BUNDLE_DIRECTORIES),
            "files": list(BUNDLE_FILES),
            "upstream": {
                "root": upstream_root if include_upstream else None,
                "included": upstream_included,
                "file_count": upstream_files,
            },
        },
        "exclusions": {
            "denied_directories": sorted(DENIED_DIR_NAMES),
            "denied_suffixes": list(DENIED_SUFFIXES),
            "denied_name_prefixes": list(DENIED_NAME_PREFIXES),
            "excluded_counts": excluded,
            "excluded_examples": excluded_examples,
        },
        "untracked_inclusion": untracked_report,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
        "artifacts": {MANIFEST_NAME: "build artifact; identical to the on-disk bundle manifest"},
    }
    manifest_path = write_json(output / MANIFEST_NAME, manifest)
    write_archive(output / ARCHIVE_NAME, source_root, sorted(files), manifest_path)
    return {
        "format": FORMAT,
        "kind": "bundle",
        "output": str(output),
        "manifest": str(manifest_path),
        "archive": str(output / ARCHIVE_NAME),
        "file_count": len(files),
        "total_bytes": manifest["total_bytes"],
        "upstream_files": upstream_files,
        "untracked_inclusion": untracked_report,
        "excluded_counts": excluded,
        "git_head": metadata.get("head"),
    }


def write_archive(
    archive_path: Path, source_root: Path, rel_paths: list[str], manifest_path: Path
) -> None:
    with archive_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz_handle:
            with tarfile.open(fileobj=gz_handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for rel in rel_paths:
                    info = archive.gettarinfo(str(source_root / rel), arcname=rel)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o644
                    with (source_root / rel).open("rb") as handle:
                        archive.addfile(info, handle)
                info = archive.gettarinfo(str(manifest_path), arcname=MANIFEST_NAME)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o644
                with manifest_path.open("rb") as handle:
                    archive.addfile(info, handle)


# --------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------


def verify_bundle(
    bundle_dir: Path,
    *,
    source_root: Path | None = None,
    protocol_path: Path | None = None,
) -> dict:
    bundle_dir = Path(bundle_dir)
    checks: list[dict] = []

    def record(check_id: str, ok: bool | None, detail: str = "") -> None:
        checks.append({"id": check_id, "ok": ok, "detail": detail})

    manifest_file = bundle_dir / MANIFEST_NAME
    if not manifest_file.is_file():
        raise PreflightUnavailable(f"bundle manifest not found: {manifest_file}")
    manifest = read_json(manifest_file)
    record("manifest:parse", isinstance(manifest, dict), str(manifest_file))
    record("manifest:format", manifest.get("format") == FORMAT, str(manifest.get("format")))
    archive_path = bundle_dir / str(manifest.get("archive", ARCHIVE_NAME))
    if not archive_path.is_file():
        record("archive:present", False, str(archive_path))
        return finish_verify(bundle_dir, manifest, checks, {}, None)
    record("archive:present", True, str(archive_path))
    archive_sha = sha256_file(archive_path)

    declared_files = manifest.get("files", {})
    members: dict[str, tarfile.TarInfo] = {}
    unsafe: list[str] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = normalize_rel(member.name)
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts:
                unsafe.append(member.name)
                continue
            members[name] = member
        record("archive:path_safety", not unsafe, f"unsafe members: {unsafe[:5]}")
        missing = sorted(set(declared_files) - set(members))
        extras = sorted(set(members) - set(declared_files) - {MANIFEST_NAME})
        record("bundle:no_missing_files", not missing, f"missing: {missing[:10]} ({len(missing)})")
        record("bundle:no_extra_files", not extras, f"extras: {extras[:10]} ({len(extras)})")
        mismatches = []
        for rel, entry in sorted(declared_files.items()):
            if rel not in members:
                continue
            handle = archive.extractfile(members[rel])
            if handle is None:
                mismatches.append({"path": rel, "reason": "unreadable-member"})
                continue
            with handle:
                computed = sha256_stream(handle)
            if computed != entry.get("sha256"):
                mismatches.append(
                    {
                        "path": rel,
                        "expected": entry.get("sha256"),
                        "computed": computed,
                    }
                )
        record(
            "bundle:hashes",
            not mismatches,
            f"mismatches: {len(mismatches)}"
            + (f", first: {mismatches[0]['path']}" if mismatches else ""),
        )
        embedded = archive.extractfile(members[MANIFEST_NAME]) if MANIFEST_NAME in members else None
        if embedded is None:
            record("bundle:embedded_manifest", False, "archive lacks bundle-manifest.json")
        else:
            with embedded:
                embedded_hash = sha256_stream(embedded)
            record(
                "bundle:embedded_manifest",
                embedded_hash == sha256_file(manifest_file),
                f"embedded={embedded_hash[:12]} on_disk={sha256_file(manifest_file)[:12]}",
            )
        denied = sorted(name for name in members if denied_reason(name))
        record("bundle:no_denied_paths", not denied, f"denied: {denied[:10]}")

    required_dirs = list(BUNDLE_DIRECTORIES)
    for rel_dir in required_dirs:
        present = any(
            name == rel_dir or name.startswith(rel_dir.rstrip("/") + "/") for name in members
        )
        record(
            f"closure:{rel_dir}",
            present,
            f"{sum(1 for name in members if name.startswith(rel_dir.rstrip('/') + '/'))} files",
        )
    for rel_file in REQUIRED_BUNDLE_FILES:
        record(f"closure:{rel_file}", rel_file in members, "")

    live_root = Path(source_root) if source_root is not None else manifest.get("source_root")
    live_root = Path(live_root) if live_root else None
    if live_root is not None and live_root.is_dir():
        drift = []
        for rel, entry in sorted(declared_files.items()):
            live = live_root / rel
            if not live.is_file():
                drift.append({"path": rel, "reason": "absent-on-source-root"})
            elif sha256_file(live) != entry.get("sha256"):
                drift.append({"path": rel, "reason": "content-changed"})
        record("source_root:no_drift", not drift, f"drift: {len(drift)}")
        metadata = git_metadata(live_root)
        untracked_missing = []
        if metadata.get("available"):
            for rel in metadata["untracked_files"]:
                in_scope = any(
                    rel == prefix or rel.startswith(prefix.rstrip("/") + "/")
                    for prefix in (*BUNDLE_DIRECTORIES, *BUNDLE_FILES)
                )
                if in_scope and rel not in declared_files and denied_reason(rel) is None:
                    untracked_missing.append(rel)
        record(
            "source_root:untracked_closure",
            not untracked_missing,
            f"untracked missing from bundle: {untracked_missing[:10]} ({len(untracked_missing)})",
        )
    else:
        record("source_root:no_drift", None, "source root unavailable; archive-only verification")

    effective_protocol = protocol_path
    if effective_protocol is None and live_root is not None:
        candidate = live_root / DEFAULT_PROTOCOL
        if candidate.is_file():
            effective_protocol = candidate
    if effective_protocol is not None:
        protocol = read_json(effective_protocol)
        missing_keys = [key for key in REQUIRED_PROTOCOL_KEYS if key not in protocol]
        record("protocol:required_keys", not missing_keys, f"missing: {missing_keys}")
        rel_protocol = DEFAULT_PROTOCOL
        if rel_protocol in declared_files and members.get(rel_protocol) is not None:
            record(
                "protocol:bundle_hash_matches_live",
                declared_files[rel_protocol]["sha256"] == sha256_file(effective_protocol),
                DEFAULT_PROTOCOL,
            )
    return finish_verify(bundle_dir, manifest, checks, members, archive_sha)


def finish_verify(
    bundle_dir: Path,
    manifest: dict,
    checks: list[dict],
    members: dict,
    archive_sha: str | None,
) -> dict:
    failures = [check for check in checks if check["ok"] is False]
    pending = [check for check in checks if check["ok"] is None]
    result = {
        "format": FORMAT,
        "kind": "verify",
        "generated_at": now_iso(),
        "bundle": str(bundle_dir),
        "archive_sha256": archive_sha,
        "files_checked": len(manifest.get("files", {})),
        "archive_members": len(members),
        "git_head": (manifest.get("git") or {}).get("head"),
        "checks": checks,
        "failures": [{"id": check["id"], "detail": check["detail"]} for check in failures],
        "pending": [check["id"] for check in pending],
        "status": "failed" if failures else ("passed_pending" if pending else "passed"),
    }
    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="owac-n01-full-preflight",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subcommands.add_parser(
        "inventory",
        help="read protocol config + split manifests + data root; emit an inventory JSON",
        description=(
            "Read-only inventory. Reads the protocol config, the four split manifests and "
            "the data root, then reports the train/calibration/validation/test summary and "
            "the hard checks (manifest presence, declared hash, non-empty, disjoint)."
        ),
        epilog=(
            "Example:\n"
            "  python -m owac.runtime.n01_full_preflight inventory \\\n"
            "      --config configs/encoders/n01_full/protocol_v1.json \\\n"
            "      --output outputs/n01-full/p0-inventory"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inventory_parser.add_argument("--config", type=Path, default=Path(DEFAULT_PROTOCOL))
    inventory_parser.add_argument("--output", type=Path, required=True)
    inventory_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    inventory_parser.add_argument(
        "--data-root", type=Path, default=None, help="override data.data_root from the protocol"
    )
    inventory_parser.add_argument(
        "--check-loader",
        action="store_true",
        help="read real batches from all four regions; requires torch",
    )
    inventory_parser.add_argument(
        "--strict", action="store_true", help="exit 1 when a check fails or stays pending"
    )

    profile_parser = subcommands.add_parser(
        "profile",
        help="drive GPU profiling collection and aggregate the raw points",
        description=(
            "Collect mode requires CUDA and refuses the local CPU fallback explicitly; "
            "summarize mode aggregates raw point JSONs on any host."
        ),
        epilog=(
            "Examples:\n"
            "  python -m owac.runtime.n01_full_preflight profile --config ... \\\n"
            "      --output ... --mode collect\n"
            "  python -m owac.runtime.n01_full_preflight profile --config ... \\\n"
            "      --output ... --mode summarize"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    profile_parser.add_argument("--config", type=Path, default=Path(DEFAULT_PROTOCOL))
    profile_parser.add_argument("--output", type=Path, required=True)
    profile_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    profile_parser.add_argument(
        "--mode",
        choices=("auto", "collect", "summarize"),
        default="auto",
        help="auto summarizes when raw points exist, otherwise collects (GPU only)",
    )
    profile_parser.add_argument("--raw-dir", type=Path, default=None)
    profile_parser.add_argument("--encoder-factory", default=None, help="module:attr override")
    profile_parser.add_argument(
        "--backbone",
        choices=PROFILING_BACKBONES,
        default=None,
        help="explicit backbone override; dbsr stays full-size (64/9/512)",
    )
    profile_parser.add_argument("--iterations", type=int, default=None)
    profile_parser.add_argument("--warmup", type=int, default=None)
    profile_parser.add_argument(
        "--io-root", type=Path, default=None, help="data root for the I/O probe"
    )
    profile_parser.add_argument("--io-sample", type=int, default=8)

    bundle_parser = subcommands.add_parser(
        "bundle",
        help="explicitly pack stage sources, configs, tests, upstream sources and locks",
        description=(
            "Walks the filesystem for the declared include paths so that new untracked "
            "sources are packed; git archive is never used. Denied paths (.venv, outputs, "
            "caches, keys) are dropped and recorded."
        ),
        epilog=(
            "Example:\n"
            "  python -m owac.runtime.n01_full_preflight bundle \\\n"
            "      --output /tmp/n01-bundle --no-include-upstream"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bundle_parser.add_argument("--output", type=Path, required=True)
    bundle_parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    bundle_parser.add_argument(
        "--include-upstream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the ignored upstream source/licence cache used by upstream_lock.json",
    )
    bundle_parser.add_argument("--upstream-root", default=DEFAULT_UPSTREAM_ROOT)

    verify_parser = subcommands.add_parser(
        "verify",
        help="hash and closure check a produced bundle (failures are exit 1)",
        description=(
            "Streams every archive member, compares it with bundle-manifest.json, checks the "
            "required closure and (when the source root is reachable) detects drift and "
            "untracked files that never made it into the bundle."
        ),
        epilog=(
            "Example:\n  python -m owac.runtime.n01_full_preflight verify --bundle /tmp/n01-bundle"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--source-root", type=Path, default=None)
    verify_parser.add_argument("--protocol", type=Path, default=None)
    verify_parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory(
                args.config,
                args.output,
                repo_root=args.repo_root,
                data_root=args.data_root,
                check_loader=args.check_loader,
            )
            emit(result)
            if args.strict and result["summary"]["status"] != "ok":
                return EXIT_FAILED
            return EXIT_OK
        if args.command == "profile":
            result = run_profile(
                args.config,
                args.output,
                repo_root=args.repo_root,
                mode=args.mode,
                raw_dir=args.raw_dir,
                factory_spec=args.encoder_factory,
                iterations=args.iterations,
                warmup=args.warmup,
                io_root=args.io_root,
                io_sample=args.io_sample,
                backbone_override=args.backbone,
            )
            emit(result)
            return EXIT_OK
        if args.command == "bundle":
            result = bundle(
                args.source_root,
                args.output,
                include_upstream=args.include_upstream,
                upstream_root=args.upstream_root,
            )
            emit(result)
            return EXIT_OK
        result = verify_bundle(
            args.bundle, source_root=args.source_root, protocol_path=args.protocol
        )
        if args.report is not None:
            write_json(args.report, result)
        emit(result)
        return EXIT_OK if result["status"] in ("passed", "passed_pending") else EXIT_FAILED
    except PreflightRefused as exc:
        emit({"status": "refused", "error": str(exc), "command": args.command})
        return EXIT_REFUSED
    except PreflightUnavailable as exc:
        emit({"status": "unavailable", "error": str(exc), "command": args.command})
        return EXIT_UNAVAILABLE
    except (OSError, ValueError, KeyError) as exc:
        emit({"status": "error", "error": f"{type(exc).__name__}: {exc}", "command": args.command})
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
