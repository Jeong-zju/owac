#!/usr/bin/env python3
"""Immutable working-tree source snapshots for N04 (S0).

Why not ``git archive HEAD``: the N04 checkout carries uncommitted source and
documents that the stages actually run.  This tool packs the *current bytes* of
every tracked file plus explicitly included untracked source, so a running
training job always corresponds to one hashed archive.

Build::

    python scripts/n04_full/package_source.py build [options]

Inputs: the git working tree at ``--repo-root`` only; no other file is read.
Outputs (three files, all inside ``--output-dir``)::

    <snapshot_id>.tar.gz
    <snapshot_id>.manifest.json
    <snapshot_id>.sha256sums.txt

with ``snapshot_id = <prefix>-<UTC timestamp>-<package sha256[:12]>`` (默认
``n04-s0``，S1 及以后用 ``--snapshot-prefix`` 改成本阶段前缀；工具本身不变).  The
archive is rooted at repo-relative paths and is byte-deterministic for a fixed
``--timestamp`` (fixed mtime, uid/gid 0, sorted members, gzip mtime 0).  Every
manifest hash is computed from the bytes written into the archive in the same
read pass, so the manifest describes the package even when another writer
touches the shared checkout mid-run; a post-build re-read reports files that
changed after packing under ``stability.unstable_files``.

Excluded: ``.git``, virtualenvs, ``outputs``, build caches, data directories and
anything above ``--max-file-bytes``.  Tracked files are otherwise packed
faithfully (a tracked file matching an artifact rule is packed and reported as a
warning); the artifact/secret rules themselves apply to the *untracked*
candidates, which are unvetted content.  Nothing is dropped silently: every
candidate path that is not packed is listed in the manifest ``excluded`` array
with a machine-readable reason.

Gates (non-zero exit, actionable message):
  * ``--require-uncommitted`` (default): at least one packed file must be
    provably absent from ``HEAD`` or modified relative to it, otherwise the
    snapshot does not represent the uncommitted work it claims to.
  * a candidate file larger than ``--max-file-bytes`` fails unless passed via
    ``--allow-large-file`` (never dropped silently).

Verify a synced tree::

    python scripts/n04_full/package_source.py verify \
        --manifest <snapshot_id>.manifest.json --tree <extracted-dir> \
        [--package <snapshot_id>.tar.gz]

``verify`` re-hashes every manifest entry and reports
``checked / mismatched / missing / extra`` counts; exit code 1 when anything is
missing or mismatched.  Extra files (e.g. ``__pycache__`` written by running the
snapshot) are reported but not fatal unless ``--strict-extras`` is given.

Resource use: each file is streamed once (hash and archive), peak memory is
bounded by the largest single file plus the manifest.  Runtime is dominated by
hashing the tree (~40 MB here).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "n04-source-snapshot-v1"
DEFAULT_SNAPSHOT_PREFIX = "n04-s0"
SNAPSHOT_PREFIX = DEFAULT_SNAPSHOT_PREFIX  # 兼容旧引用
SNAPSHOT_ID_RE = re.compile(r"^n04-[a-z0-9]+-\d{8}T\d{6}Z-[0-9a-f]{12}$")
DEFAULT_UNTRACKED_ROOTS = ("src", "tests", "configs", "scripts", "docs", "research")
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024

DENY_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "outputs",
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
        ".cache",
        ".tox",
        ".eggs",
        "node_modules",
        "site-packages",
        ".ssh",
        ".aws",
    }
)
DENY_DIR_GLOBS = ("*.egg-info",)
DENY_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".so",
    ".o",
    ".a",
    ".dylib",
    ".dll",
    ".whl",
    ".egg",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".npz",
    ".npy",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pkl",
    ".pickle",
    ".bin",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".parquet",
    ".feather",
    ".arrow",
    ".sqlite",
    ".sqlite3",
    ".db",
)
DENY_NAMES = frozenset(
    {
        ".env",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "known_hosts",
    }
)
DENY_NAME_PREFIXES = (".env.",)
DENY_NAME_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".token")


class PackagingError(RuntimeError):
    """A gate or environment problem the caller must fix."""


@dataclass(frozen=True)
class Entry:
    path: str
    sha256: str
    size: int
    mode: str
    origin: str
    large_file_override: bool = False

    def as_json(self) -> dict:
        out = {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.size,
            "mode": self.mode,
            "origin": self.origin,
        }
        if self.large_file_override:
            out["large_file_override"] = True
        return out


@dataclass(frozen=True)
class StagedFile:
    """A candidate accepted for packing, before its bytes are read."""

    path: str
    size: int
    mode: str
    origin: str
    large_file_override: bool = False


def _git(repo: Path, *args: str, check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
    )
    if check and proc.returncode != 0:
        raise PackagingError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def _split_z(raw: bytes) -> list[str]:
    return [chunk.decode("utf-8", "surrogateescape") for chunk in raw.split(b"\0") if chunk]


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


class _HashingReader:
    """Read-through wrapper that hashes exactly the bytes handed to the archive.

    The manifest hash must describe the bytes inside the tar, not a separate
    earlier read: on a live working tree another process can rewrite a file
    between two passes, which would make the archive and its manifest disagree.
    """

    def __init__(self, handle) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        block = self._handle.read(size)
        if block:
            self.digest.update(block)
            self.count += len(block)
        return block


def dir_deny_reason(rel_path: str) -> str | None:
    """Generated/foreign directory that must never enter a snapshot."""
    for part in rel_path.split("/")[:-1]:
        if part in DENY_DIRS:
            return f"deny-dir:{part}"
        if any(fnmatch.fnmatch(part, pattern) for pattern in DENY_DIR_GLOBS):
            return f"deny-dir-glob:{part}"
    return None


def content_deny_reason(rel_path: str) -> str | None:
    """Artifact/secret-like path, applied to untracked candidates only."""
    name = rel_path.split("/")[-1]
    if name in DENY_NAMES:
        return f"deny-name:{name}"
    if name.startswith(DENY_NAME_PREFIXES) and name != ".env.example":
        return f"deny-name-prefix:{name}"
    lowered = name.lower()
    for suffix in DENY_NAME_SUFFIXES:
        if lowered.endswith(suffix):
            return f"deny-secret-suffix:{suffix}"
    lowered_path = rel_path.lower()
    for suffix in DENY_SUFFIXES:
        if lowered_path.endswith(suffix):
            return f"deny-suffix:{suffix}"
    return None


def untracked_scope_reason(rel_path: str, untracked_roots: tuple[str, ...]) -> str | None:
    """Untracked files are packed only from explicitly included top-level roots."""
    if rel_path.split("/")[0] not in untracked_roots:
        return f"untracked-out-of-scope:{rel_path.split('/')[0]}"
    return None


def _parse_status(repo: Path) -> list[dict]:
    raw = _git(repo, "status", "--porcelain=v1", "-z")
    tokens = _split_z(raw)
    entries: list[dict] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if len(token) < 3:
            continue
        state, path = token[:2], token[3:]
        record = {"state": state, "path": path}
        if state[0] in "RC":
            index += 1  # original path of a rename/copy
        entries.append(record)
    return entries


def collect_candidates(repo: Path) -> tuple[list[tuple[str, str]], list[str]]:
    tracked = _split_z(_git(repo, "ls-files", "-z"))
    untracked = _split_z(_git(repo, "ls-files", "-z", "--others", "--exclude-standard"))
    return [(p, "tracked") for p in tracked], untracked


def _tracked_modes(repo: Path) -> dict[str, str]:
    modes: dict[str, str] = {}
    for line in _split_z(_git(repo, "ls-files", "-s", "-z")):
        meta, _, path = line.partition("\t")
        if not path:
            continue
        modes[path] = meta.split()[0]
    return modes


def _gather_entries(
    repo: Path,
    candidates: list[tuple[str, str]],
    untracked_roots: tuple[str, ...],
    max_bytes: int,
    allow_large: frozenset[str],
) -> tuple[list[StagedFile], list[dict], list[dict]]:
    modes = _tracked_modes(repo)
    staged: list[StagedFile] = []
    excluded: list[dict] = []
    warnings: list[dict] = []
    for rel_path, origin in candidates:
        full = repo / rel_path
        if full.is_symlink():
            raise PackagingError(f"refusing to pack symlink without explicit handling: {rel_path}")
        if not full.is_file():
            excluded.append({"path": rel_path, "origin": origin, "reason": "not-a-regular-file"})
            continue
        size = full.stat().st_size
        reason = dir_deny_reason(rel_path)
        if reason is None and origin == "untracked":
            reason = untracked_scope_reason(rel_path, untracked_roots) or content_deny_reason(
                rel_path
            )
        if reason is None and size > max_bytes and rel_path not in allow_large:
            raise PackagingError(
                f"{rel_path} is {size} bytes, above --max-file-bytes={max_bytes}; "
                "it is a candidate source file and is not dropped silently: pack it "
                f"with --allow-large-file {rel_path!r} or raise --max-file-bytes"
            )
        override = size > max_bytes and rel_path in allow_large
        if reason is not None:
            excluded.append({"path": rel_path, "origin": origin, "reason": reason, "bytes": size})
            continue
        if origin == "tracked":
            flagged = content_deny_reason(rel_path)
            if flagged is not None:
                warnings.append({"path": rel_path, "warning": f"tracked-content-rule:{flagged}"})
        if origin == "tracked":
            raw_mode = modes.get(rel_path, "100644")
        else:
            raw_mode = "100755" if os.access(full, os.X_OK) else "100644"
        staged.append(
            StagedFile(
                path=rel_path,
                size=size,
                mode=raw_mode,
                origin=origin,
                large_file_override=override,
            )
        )
    staged.sort(key=lambda item: item.path)
    excluded.sort(key=lambda item: item["path"])
    warnings.sort(key=lambda item: item["path"])
    return staged, excluded, warnings


def build_archive(
    repo: Path, staged: list[StagedFile], epoch: int, out_path: Path
) -> tuple[str, list[Entry]]:
    """Write a deterministic tar.gz, hashing each file in the same read.

    Returns ``(package_sha256, entries)`` where every entry hash covers exactly
    the bytes stored in the archive.
    """
    entries: list[Entry] = []
    with open(out_path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for item in staged:
                    info = tarfile.TarInfo(item.path)
                    info.size = item.size
                    info.mtime = epoch
                    info.mode = 0o755 if item.mode == "100755" else 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.type = tarfile.REGTYPE
                    with open(repo / item.path, "rb") as handle:
                        reader = _HashingReader(handle)
                        tar.addfile(info, reader)
                    entries.append(
                        Entry(
                            path=item.path,
                            sha256=reader.digest.hexdigest(),
                            size=reader.count,
                            mode=item.mode,
                            origin=item.origin,
                            large_file_override=item.large_file_override,
                        )
                    )
    digest, _ = sha256_file(out_path)
    return digest, entries


def _parse_timestamp(value: str | None) -> _dt.datetime:
    if value is None:
        return _dt.datetime.now(_dt.UTC)
    text = value.strip()
    if text.isdigit():
        return _dt.datetime.fromtimestamp(int(text), _dt.UTC)
    if re.fullmatch(r"\d{8}T\d{6}Z", text):
        return _dt.datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=_dt.UTC)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = _dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return parsed.astimezone(_dt.UTC)


def _head_paths(repo: Path) -> set[str]:
    raw = _git(repo, "ls-tree", "-r", "--name-only", "-z", "HEAD", check=False)
    return set(_split_z(raw))


def cmd_build(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    root = _git(repo, "rev-parse", "--show-toplevel", check=False).decode().strip()
    if not root:
        raise PackagingError(f"{repo} is not inside a git working tree")
    repo = Path(root)
    head = _git(repo, "rev-parse", "HEAD", check=False).decode().strip() or "UNBORN"
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).decode().strip()
    status = _parse_status(repo)
    diff_raw = _git(repo, "diff", "--binary", "HEAD", check=False)
    head_paths = _head_paths(repo) if head != "UNBORN" else set()

    tracked, untracked = collect_candidates(repo)
    roots = tuple(args.include_untracked_root)
    staged, excluded, warnings = _gather_entries(
        repo,
        tracked + [(p, "untracked") for p in untracked],
        roots,
        args.max_file_bytes,
        frozenset(args.allow_large_file),
    )

    packed = {item.path for item in staged}
    modified_tracked = [
        item["path"] for item in status if item["state"] not in ("??",) and item["path"] in packed
    ]
    untracked_not_in_head = sorted(p for p in packed if p not in head_paths and p in set(untracked))
    uncommitted = sorted(set(modified_tracked) | set(untracked_not_in_head))
    evidence = {
        "required": bool(args.require_uncommitted),
        "verified": bool(uncommitted),
        "modified_tracked": sorted(modified_tracked),
        "untracked_not_in_head": untracked_not_in_head,
        "head_file_count": len(head_paths),
    }
    if args.require_uncommitted and not uncommitted:
        excluded_uncommitted = sorted(item["path"] for item in status if item["path"] not in packed)
        detail = (
            f"; {len(excluded_uncommitted)} uncommitted path(s) were excluded: "
            f"{excluded_uncommitted[:5]}"
            if excluded_uncommitted
            else ""
        )
        raise PackagingError(
            "no uncommitted source detected in the packed set: a HEAD-only snapshot "
            "does not prove the working tree content; pass --no-require-uncommitted "
            f"only when the tree really is clean{detail}"
        )

    epoch_dt = _parse_timestamp(args.timestamp)
    epoch = int(epoch_dt.timestamp())
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    staging = out_dir / f".staging-{os.getpid()}.tar.gz"
    try:
        package_sha, entries = build_archive(repo, staged, epoch, staging)
        prefix = getattr(args, "snapshot_prefix", DEFAULT_SNAPSHOT_PREFIX)
        snapshot_id = f"{prefix}-{epoch_dt.strftime('%Y%m%dT%H%M%SZ')}-{package_sha[:12]}"
        package_path = out_dir / f"{snapshot_id}.tar.gz"
        os.replace(staging, package_path)
    finally:
        if staging.exists():
            staging.unlink()

    if uncommitted:
        sample_path = sorted(untracked_not_in_head)[0] if untracked_not_in_head else uncommitted[0]
        sample = next(entry for entry in entries if entry.path == sample_path)
        evidence["sample"] = {
            "path": sample.path,
            "sha256": sample.sha256,
            "origin": sample.origin,
            "present_in_head": sample.path in head_paths,
            "present_in_manifest": sample.path in packed,
        }

    # The tree can move while this runs (other agents share the checkout).  Re-read
    # every packed file and compare with the archived bytes: the snapshot is only
    # reported as stable when the working tree still matches what was packed.
    status_after = _parse_status(repo)
    diff_after = _git(repo, "diff", "--binary", "HEAD", check=False)
    unstable = []
    for entry in entries:
        target = repo / entry.path
        if not target.is_file():
            unstable.append({"path": entry.path, "reason": "removed-after-packing"})
            continue
        digest, size = sha256_file(target)
        if digest != entry.sha256 or size != entry.size:
            unstable.append(
                {"path": entry.path, "reason": "rewritten-after-packing", "bytes_now": size}
            )
    stability = {
        "stable": not unstable,
        "unstable_files": unstable,
        "status_changed": status_after != status,
        "tracked_diff_changed": hashlib.sha256(diff_after).hexdigest()
        != hashlib.sha256(diff_raw).hexdigest(),
        "note": (
            "hashes describe the bytes stored in the archive (single read pass); "
            "unstable_files lists working-tree files that changed after packing"
        ),
    }

    manifest = {
        "schema": SCHEMA,
        "snapshot_id": snapshot_id,
        "created_utc": epoch_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "package": {
            "path": package_path.name,
            "sha256": package_sha,
            "bytes": package_path.stat().st_size,
            "format": "tar.gz",
            "deterministic_epoch": epoch,
            "contents_root": "repo-relative (extract into source/<snapshot_id>/)",
        },
        "git": {
            "repo_root": str(repo),
            "head": head,
            "branch": branch,
            "dirty": bool(status),
            "status": status,
            "tracked_diff_sha256": hashlib.sha256(diff_raw).hexdigest(),
            "tracked_diff_bytes": len(diff_raw),
        },
        "tool": {
            "name": "scripts/n04_full/package_source.py",
            "sha256": sha256_file(Path(__file__).resolve())[0],
            "python": sys.version.split()[0],
        },
        "include_untracked_roots": sorted(roots),
        "counts": {
            "packed": len(entries),
            "packed_tracked": sum(1 for e in entries if e.origin == "tracked"),
            "packed_untracked": sum(1 for e in entries if e.origin == "untracked"),
            "excluded": len(excluded),
            "excluded_tracked": sum(1 for e in excluded if e["origin"] == "tracked"),
            "packed_bytes": sum(e.size for e in entries),
        },
        "uncommitted_evidence": evidence,
        "stability": stability,
        "files": [entry.as_json() for entry in entries],
        "excluded": excluded,
        "warnings": warnings,
    }
    manifest_path = out_dir / f"{snapshot_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", "utf-8")

    sums_path = out_dir / f"{snapshot_id}.sha256sums.txt"
    with open(sums_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"{entry.sha256}  {entry.path}\n")

    print(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "package_sha256": package_sha,
                "package_bytes": manifest["package"]["bytes"],
                "package_path": str(package_path),
                "manifest_path": str(manifest_path),
                "sha256sums_path": str(sums_path),
                "head": head,
                "branch": branch,
                "tracked_diff_sha256": manifest["git"]["tracked_diff_sha256"],
                "counts": manifest["counts"],
                "excluded": [item["path"] for item in excluded],
                "warnings": warnings,
                "stability": stability,
                "uncommitted_evidence": {
                    "verified": evidence["verified"],
                    "sample": evidence.get("sample"),
                    "modified_tracked": len(evidence["modified_tracked"]),
                    "untracked_not_in_head": len(evidence["untracked_not_in_head"]),
                },
            },
            indent=2,
        )
    )
    return 0


def _is_allowed_extra(rel_path: str, patterns: tuple[str, ...]) -> bool:
    parts = rel_path.split("/")
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in parts[:-1]):
            return True
    return False


def cmd_verify(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise PackagingError(f"unexpected manifest schema: {manifest.get('schema')!r}")
    tree = Path(args.tree).resolve()
    if not tree.is_dir():
        raise PackagingError(f"--tree is not a directory: {tree}")

    files = manifest.get("files", [])
    mismatched: list[dict] = []
    missing: list[str] = []
    expected: set[str] = set()
    for record in files:
        rel = record["path"]
        expected.add(rel)
        target = tree / rel
        if not target.is_file():
            missing.append(rel)
            continue
        actual, size = sha256_file(target)
        if actual != record["sha256"] or size != record["bytes"]:
            mismatched.append(
                {"path": rel, "expected": record["sha256"], "actual": actual, "bytes": size}
            )

    extras = [
        p.relative_to(tree).as_posix()
        for p in sorted(tree.rglob("*"))
        if p.is_file() and p.relative_to(tree).as_posix() not in expected
    ]
    extras = [rel for rel in extras if not _is_allowed_extra(rel, tuple(args.allow_extra))]

    package_match: bool | None = None
    if args.package:
        actual_pkg, _ = sha256_file(Path(args.package))
        package_match = actual_pkg == manifest["package"]["sha256"]

    differences = len(mismatched) + len(missing)
    report = {
        "snapshot_id": manifest["snapshot_id"],
        "manifest": str(manifest_path),
        "tree": str(tree),
        "checked": len(files),
        "mismatched_count": len(mismatched),
        "missing_count": len(missing),
        "extra_count": len(extras),
        "differences": differences,
        "mismatched": mismatched[:20],
        "missing": missing[:20],
        "extra": extras[:40],
        "package_sha256_match": package_match,
    }
    print(json.dumps(report, indent=2))
    if package_match is False:
        return 1
    if differences:
        return 1
    if extras and args.strict_extras:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="package_source.py",
        description=__doc__.split("\n", 1)[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="pack the current working tree")
    build.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="checkout to pack (default: the checkout containing this script)",
    )
    build.add_argument(
        "--output-dir",
        default="outputs/n04-v1/s0/source-packages",
        help="directory for tar.gz + manifest + sha256sums",
    )
    build.add_argument(
        "--snapshot-prefix",
        default=DEFAULT_SNAPSHOT_PREFIX,
        help="snapshot id 前缀（默认 n04-s0；阶段工具可用 n04-s1 等）",
    )
    build.add_argument(
        "--timestamp",
        default=None,
        help="UTC timestamp (ISO-8601, YYYYMMDDTHHMMSSZ or epoch); fixes the archive",
    )
    build.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help=f"per-file size gate (default {DEFAULT_MAX_FILE_BYTES})",
    )
    build.add_argument(
        "--allow-large-file",
        action="append",
        default=[],
        metavar="PATH",
        help="explicitly allow one over-size file (repeatable)",
    )
    build.add_argument(
        "--include-untracked-root",
        action="append",
        default=list(DEFAULT_UNTRACKED_ROOTS),
        metavar="TOP",
        help="top-level directory whose untracked files are packed (repeatable)",
    )
    gate = build.add_mutually_exclusive_group()
    gate.add_argument(
        "--require-uncommitted",
        dest="require_uncommitted",
        action="store_true",
        default=True,
        help="fail unless a packed file is provably uncommitted (default)",
    )
    gate.add_argument(
        "--no-require-uncommitted",
        dest="require_uncommitted",
        action="store_false",
        help="allow a HEAD-only snapshot (use only for a genuinely clean tree)",
    )
    build.set_defaults(func=cmd_build)

    verify = sub.add_parser("verify", help="re-hash an extracted snapshot tree")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--tree", required=True)
    verify.add_argument("--package", default=None, help="also check this tar.gz sha256")
    verify.add_argument(
        "--allow-extra",
        action="append",
        default=["*.pyc", "__pycache__", ".pytest_cache", ".ruff_cache"],
        metavar="GLOB",
        help="extra files that are reported but never fatal (repeatable)",
    )
    verify.add_argument("--strict-extras", action="store_true")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PackagingError as exc:
        print(f"package_source.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
