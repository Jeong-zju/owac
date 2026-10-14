# N02 full (v1.1) encoder configs

`p0.json` is the P0 configuration: the ConvNeXt V2-Base RAW observation model, the
world/entity capacity, the three declared seeds and the data/runtime/profile blocks.
It is loaded by `owac.frameworks.n02_full.config.load_config`, which keeps declared
defaults for missing keys and rejects unknown top-level keys or type mismatches with
the offending key in the message. `config_hash` is the sha256 of the canonical JSON
(sort_keys) of the resolved config and is what stage decisions and run manifests
record. `p0.json` currently resolves to exactly `N02FullConfig()`; the semantic
checks (`validate_config`) are empty for it.

| Field | Meaning |
| --- | --- |
| `version` | model-line version, not the config schema (`CONFIG_VERSION = n02-full-p0-v1`) |
| `seeds` | `1103, 2207, 3301`; frozen in P0 and not reselected after results |
| `worlds` / `slots` / `slot_dim` | world candidates `K=8`, active slots `N=16`, width `D=512` |
| `depths` / `dims` | official ConvNeXt V2-Base `[3, 3, 27, 3]` / `[128, 256, 512, 1024]` |
| `input_size`, `scales` | 256x256 input, multi-scale `stride8`/`stride16` retained |
| `partitions` | train/dev/calibration weights summing to 1; the sealed test set is a separate manifest |
| `data` | audited dataset root and format; both stay `null` until a real RawVLA NPZ root is reachable |
| `runtime` | dtype (`float32`/`float16`/`bfloat16`), device and AMP flag |
| `profile` | profiling budget: `updates=200`, `warmup_updates=20`, batch and sequence length |
| `paths` | `output_root` for every stage artifact |

`data.notes` is intentionally empty so the shipped file equals the declared defaults and
its `config_hash` is reproducible. Environment observations belong in the stage JSON
under `outputs/`, not in the tracked config.

## P0 CLI

```bash
python -m owac.evaluation.n02_full.p0 inventory --config configs/encoders/n02_full/p0.json --output outputs/n02-full/p0/inventory
python -m owac.evaluation.n02_full.p0 parity    --config configs/encoders/n02_full/p0.json --output outputs/n02-full/p0/parity
python -m owac.evaluation.n02_full.p0 profile   --config configs/encoders/n02_full/p0.json --output outputs/n02-full/p0/profile
python -m owac.evaluation.n02_full.p0 report    --config configs/encoders/n02_full/p0.json --output outputs/n02-full/p0/report
```

Every subcommand writes one JSON document (`<output>/<subcommand>.json`); `report` writes
`report.json`, `metrics.json`, `run_manifest.json` and `stage_decision.json` from the
`inventory`/`parity`/`profile` evidence. Missing data roots, missing upstream modules and
upstream call failures are reported as `status="not_run"` with a reason and exit code 0;
nothing unverified is written as `pass`. The profile output always has three explicit
columns: `measured`, `composed_estimate`, `not_measured`. The profile target protocol
(what each entry returned by `profile_targets.build_profile_targets` must expose) is
documented in the `p0.py` module docstring.

## Ownership

- `p0.json`, this README and `tests/encoders/n02_full/` belong to the P0 foundation task.
- `third_party.lock.json` (upstream repo/commit/file hashes, licenses and local fetch
  locations) is owned by the upstream-parity task (`n02-parity`); the `inventory`
  subcommand only reads it and records a summary (entry count, per-entry commit/hash
  fields, file and canonical hashes), never its full content.

## Current P0 evidence and P1 data use

The historical first pass is superseded by
[the independent acceptance record](../../../research/experiments/2026-10-14_n02_p0_independent_acceptance.md).
Real cache checks ran on the data host, including actual checkpoint normalization.
Use the v2 manifests and frozen anchor tables identified by `frozen-data.json` in
that record. Old task/global manifests have different membership and are not the
P1 split.

`p0-remote-libero.json` and `p0-remote-robotwin.json` point at the actual host paths.
Use `load_formal_normalisation(dataset_format, cfg.data["stats_path"])` and pass
that result explicitly as `normalisation=` to `ManifestWindowDataset`. The generic
loader defaults are preserved for old callers; they are not the formal checkpoint
normalization. The frozen anchor table supplies the complete window and conflicting
window overrides raise an error.

`profile.operation=train_step` measures real optimizer updates; the separate
`p0-forward-profile.json` measures inference forwards. `runtime.acceptance_budgets`
freezes a 200ms decision p95 and 256MiB persistent-state development budget. These
are future integrated-system bounds, not already-passed full-model measurements.
