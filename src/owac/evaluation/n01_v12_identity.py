"""Content-verified demonstration -> actual initial-state mappings.

No assumption that trial_idx == demo_idx or demo_idx == init_state_index. Missing
source data produces no verified mapping. Ambiguous duplicate initial states are
retained as ambiguity, never resolved by choosing the matching numeric index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def content_digest(actions, first_state):
    digest = hashlib.sha256()
    for value in (actions, first_state):
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def verify_demo_content(
    cached_actions, cached_states, source_actions, source_states, init_states, *, atol=1e-6
):
    arrays = [
        np.asarray(x)
        for x in (cached_actions, cached_states, source_actions, source_states, init_states)
    ]
    actions, states, original_actions, original_states, initial = arrays
    if any(not np.isfinite(x).all() for x in arrays) or not len(actions) or not len(states):
        return {"verified": False, "reason": "empty or non-finite source contents"}
    if (
        actions.ndim != 2
        or original_actions.ndim != 2
        or actions.shape[1:] != original_actions.shape[1:]
    ):
        return {"verified": False, "reason": "action shape mismatch"}
    if (
        len(states) != len(actions)
        or states.ndim != 2
        or original_states.ndim != 2
        or len(original_states) < len(original_actions)
        or states.shape[1:] != original_states.shape[1:]
    ):
        return {"verified": False, "reason": "state shape mismatch"}
    # Match action and state jointly. Repeated no-op actions must not select the
    # wrong source position before a later state-content comparison.
    indices = []
    cursor = 0
    for action, state in zip(actions, states, strict=True):
        while cursor < len(original_actions):
            if np.allclose(action, original_actions[cursor], atol=atol, rtol=0) and np.allclose(
                state, original_states[cursor], atol=atol, rtol=0
            ):
                break
            cursor += 1
        if cursor == len(original_actions):
            return {
                "verified": False,
                "reason": "cache action/state pairs are not a source subsequence",
            }
        indices.append(cursor)
        cursor += 1
    if initial.ndim != 2 or initial.shape[1:] != original_states.shape[1:]:
        return {"verified": False, "reason": "initial-state vector shape mismatch"}
    matches = np.flatnonzero(
        np.all(np.isclose(initial, original_states[0], atol=atol, rtol=0), axis=1)
    ).tolist()
    return {
        "verified": len(matches) == 1,
        "reason": "content_match"
        if len(matches) == 1
        else "missing or ambiguous initial-state content match",
        "matching_init_state_indices": matches,
        "matched_action_indices": indices,
        "cache_content_digest": content_digest(actions, states),
    }


def verified_split_index(manifest_path, raw_root, proof_path):
    manifest = json.loads(Path(manifest_path).read_text())
    proof = json.loads(Path(proof_path).read_text())
    if proof.get("schema") != "n01-v12-content-identity-v1":
        raise ValueError("a content-verification receipt is required")
    rows = {r["relative_path"]: r for r in manifest["rows"]}
    index = {}
    for row in proof["rows"]:
        if not row.get("verified"):
            continue
        relative = row["relative_path"]
        if relative not in rows:
            raise ValueError("identity receipt references an episode outside the manifest")
        with np.load(Path(raw_root) / relative, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            digest = content_digest(data["actions"], data["states"])
        if digest != row["cache_content_digest"]:
            raise ValueError("cache content changed after identity verification")
        if (
            row.get("demo_index") != int(metadata["demo_index"])
            or row.get("task_id") != int(metadata["task_id"])
            or row.get("suite") != str(metadata["suite"])
        ):
            raise ValueError("identity receipt metadata does not match the current cache")
        matches = row.get("matching_init_state_indices", [])
        if len(matches) != 1:
            raise ValueError("verified identity has ambiguous initial-state content")
        key = (str(row["suite"]), int(row["task_id"]), int(matches[0]))
        side = str(rows[relative]["split"])
        if key in index and index[key] != side:
            raise ValueError("the same physical initial state crosses split boundaries")
        index[key] = side
    return index


def audit(args):
    manifest = json.loads(Path(args.manifest).read_text())
    results = []
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 identity audit needs the optional 'identity' dependency") from exc
    import torch

    for row in manifest["rows"]:
        relative = row["relative_path"]
        with np.load(Path(args.raw_root) / relative, allow_pickle=False) as cache:
            meta = json.loads(str(cache["metadata_json"]))
            suite = str(meta["suite"])
            task = str(meta.get("task_name", ""))
            result = {
                "relative_path": relative,
                "suite": suite,
                "task_id": int(meta["task_id"]),
                "demo_index": int(meta["demo_index"]),
                "split": row["split"],
                "verified": False,
            }
            hdf5 = Path(args.hdf5_root) / suite / f"{task}_demo.hdf5"
            initial = Path(args.init_root) / suite / f"{task}.pruned_init"
            if not task or not hdf5.is_file() or not initial.is_file():
                result["reason"] = "original demonstration or initial-state file missing"
            else:
                init_states = torch.load(initial, map_location="cpu", weights_only=False)
                with h5py.File(hdf5, "r") as source:
                    key = f"demo_{int(meta['demo_index'])}"
                    if key not in source["data"]:
                        result["reason"] = "declared demonstration missing in source"
                    else:
                        demo = source["data"][key]
                        result.update(
                            verify_demo_content(
                                cache["actions"],
                                cache["states"],
                                demo["actions"][()],
                                demo["states"][()],
                                np.asarray(init_states),
                                atol=args.atol,
                            )
                        )
            results.append(result)
    receipt = {
        "schema": "n01-v12-content-identity-v1",
        "atol": args.atol,
        "verified": sum(r["verified"] for r in results),
        "total": len(results),
        "rows": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "raw-root", "hdf5-root", "init-root", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args(argv)
    report = audit(args)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    return 0 if report["verified"] == report["total"] and report["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
