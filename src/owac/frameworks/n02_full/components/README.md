# N02 full — ported upstream components and parity receipts (P0)

> 2026-10-14 independent repair: all 17 reference checks pass, including official custom sparse FCMAE, SAVi Processor/predictor gradients, official Theseus and Gen. The dense sparse-math port was corrected; it still has different computational cost. See `research/experiments/2026-10-14_n02_p0_independent_acceptance.md` and the new result directory. Sections below preserve first-pass provenance; their missing-reference entries are historical.

Owner: `n02-parity`.  Everything in this directory is either a **verbatim upstream copy**
(`upstream/`, `licenses/`) or a **port written here**; the distinction is enforced by
`configs/encoders/n02_full/third_party.lock.json` (`vendored_path`, `port_diff`, `status`) and by
`outputs/n02-full/p0/parity/logs/22_vendored_vs_cache_sha256.txt`.

## 1. Layout

| Path | What it is |
| --- | --- |
| `upstream/` | verbatim upstream sources (`upstream/convnextv2`, `upstream/savi`, `upstream/dnri`, `upstream/theseus`, `upstream/particles`, `upstream/genjl`) |
| `licenses/` | the upstream `LICENSE` file of every repository referenced by the lock |
| `convnextv2.py` | RAW-adapted dense ConvNeXt V2 (Base `[3,3,27,3]`/`[128,256,512,1024]`, multichannel, rectangular, multiscale stride 4/8/16/32) |
| `fcmae.py` | FCMAE with the channel/rectangle fixes, the mask-aware dense encoder port and the plain zero-fill control |
| `savi.py` | SAVi/Flax → PyTorch port (SlotAttention, inverted attention, GRU, MLP, Processor) + the masked-attention extension |
| `dnri.py` | dNRI encoder/decoder port with the online prior-only path |
| `theseus_ref.py` | hand-written LM reference + full-objective acceptance (not upstream code) |
| `smc_ref.py` | upstream `particles` bootstrap filter vs exact filtering on a fixed small HMM |
| `upstream_refs.py` | the **only** module that imports upstream code; runs it verbatim and exports reference fixtures |
| `parity_suite.py` | frozen parity interface (`PARITY_VERSION = "n02-full-components-v1"`) |
| `profile_targets.py` | profile targets for the P0 resource measurements |

## 2. Provenance and the 52-file audit set

* The audit set is `outputs/n02-v11-prompt-audit-20261013/retrieved-sources.json` (52 files,
  retrieved 2026-10-13).  On 2026-10-14 every cached file was re-hashed: **52/52 match** the
  manifest (`logs/00_sha256_verification.txt`).
* The same 52 files were compared against a fresh download of the **pinned commit** tarball
  (`https://codeload.github.com/<repo>/tar.gz/<commit>`, no `latest`): **52/52 byte-identical**
  (`logs/20_pinned_tarball_fetch.log`, `logs/21_cached_vs_pinned_commit_verification.txt`).  The
  complete pinned trees live in the ignored `third_party/pinned/` and are what the reference runs
  import.
* Vendored files are byte-identical to the cache: 22/22 (`logs/22_vendored_vs_cache_sha256.txt`).
* Six *supplementary* files were added because the cached modules import them; they come from the
  same pinned commits and are listed in `supplementary_entries` of the lock.  Two of them are
  intentionally **reduced variants** (marked `EDITED` in the hash log and described in the lock):
  `savi/lib/utils.py` keeps only the five helpers the cached modules use (the upstream file pulls
  in tensorflow/clu/matplotlib/skimage for logging helpers never called here), and
  `savi/modules/__init__.py` is empty instead of re-exporting ~12 modules outside the pinned set.

## 3. Reference environment

`outputs/n02-full/p0/parity-venv` (created 2026-10-14, `outputs/` is git-ignored).  Install logs:
`logs/10_parity_venv_jax_install.log`, `logs/12_parity_venv_torch_timm_install.log`,
`logs/16_numba_install.log`.

| package | version | used for |
| --- | --- | --- |
| python | 3.12.12 | |
| torch | 2.9.1+cpu | upstream ConvNeXt V2, dNRI, port comparison |
| timm | 1.0.20 (installed `--no-deps` + torchvision 0.24.1+cpu) | upstream `models/convnextv2.py` imports |
| jax / jaxlib | 0.4.35 | upstream SAVi (Flax) reference export |
| flax | 0.10.4 | idem; `upstream_refs.patch_flax_grucell` restores the removed default `features` field of `linen.GRUCell` that upstream `misc.GRU` relies on |
| numba | 0.61.2 | `particles.resampling` imports it at module scope |
| joblib | required by `particles.utils` | |

Attempts that failed and are recorded verbatim:
`logs/13_theseus_install_attempt.log` (theseus-ai),
`logs/14_minkowski_install_attempt.log` (MinkowskiEngine; note `/usr/bin/nvcc` exists but
`CUDA_HOME` is unset).

## 4. Frozen interfaces

### 4.1 parity suite (`parity_suite.py`)

```python
PARITY_VERSION = "n02-full-components-v1"
available_checks() -> tuple[str, ...]
run_all(config=None, *, device="cpu", dtype="float32", checks=None) -> dict
```

`run_all` never raises.  The payload is
`{"parity_version", "status", "checks", "pass_count", "fail_count", "not_run_count",
"tolerances", "versions", "paths", "device", "dtype", "elapsed_seconds", "run_at"}` where each
entry of `checks` is
`{"status", "max_abs_err", "max_rel_err", "atol", "rtol", "reason", "evidence_path", "metrics"}`.
A check whose dependency or reference fixture is missing is `not_run` with a specific reason and
must never be reported as a pass.  `config` accepts `{"fixture_dir": ..., "output_dir": ...}`.
Tolerances (P0 contract section 5): forward `atol=1e-5, rtol=1e-4`; input/parameter gradients
`atol=1e-5, rtol=1e-3` (OR-rule per tensor); FP64 analytic solutions `<= 1e-8` with numerical
gradients reported as a step sweep.

### 4.2 profile targets (`profile_targets.py`)

```python
build_profile_targets(config) -> dict[str, torch.nn.Module]
profile_target_specs(config)  -> dict[str, dict]      # {"module", "batch_size", ...}
```

Every value of `build_profile_targets` is a real `torch.nn.Module` that can be called directly, and
additionally carries `profile_meta = {"batch_size", "sequence_length", "input_shape", "kind",
"notes", "params"}`.  A profiler therefore only needs
`module.profile_meta["batch_size"]` / `["input_shape"]`; `profile_target_specs` exists for callers
that prefer an explicit description.  `torch` is imported lazily inside the builders.

Targets: `convnext_raw4` and `convnext_raw3` (Base depths `[3,3,27,3]`, dims
`[128,256,512,1024]`, 256x256 by default; 87 694 848 and 87 692 800 parameters), `fcmae_patchify_unpatchify`,
`fcmae_mask_decoder`, `fcmae_full` (92 449 792 parameters at the default widths).
`config` accepts `{"profile": {"img_size": (h, w), "batch_size": n, ...}}`.

## 5. How to reproduce

```bash
# 1. reference fixtures (isolated venv, upstream code verbatim)
PYTHONPATH=src outputs/n02-full/p0/parity-venv/bin/python -m \
  owac.frameworks.n02_full.components.upstream_refs --check all \
  --out outputs/n02-full/p0/parity/fixtures --report outputs/n02-full/p0/parity/upstream_run_report.json

# 2. SMC reference (needs numba, i.e. the isolated venv)
PYTHONPATH=src outputs/n02-full/p0/parity-venv/bin/python -m \
  owac.frameworks.n02_full.components.smc_ref --out outputs/n02-full/p0/parity/smc

# 3. parity suite against the exported fixtures (root venv is enough)
.venv/bin/python -m owac.frameworks.n02_full.components.parity_suite \
  --output outputs/n02-full/p0/parity --fixtures outputs/n02-full/p0/parity/fixtures

# 4. tests
.venv/bin/python -m pytest tests/encoders/n02_full -q -k "components or parity"
```

## 6. Status summary (2026-10-14)

Reference-venv parity receipt: `outputs/n02-full/p0/parity/parity_reference_venv.json`
(11 pass / 0 fail / 5 not_run).  Root-venv receipt: `outputs/n02-full/p0/parity/parity.json`
(`smc_ref` is `not_run` there only because the root environment has no numba; the measured receipt
is `outputs/n02-full/p0/parity/smc/smc_reference.json`).

`not_run` items and their evidence: upstream sparse FCMAE encoder (MinkowskiEngine, log 14), SAVi
`Processor` (flax `nn.scan` vs the reference jax, log 30), theseus-ai (log 13), Gen.jl (Julia not
installed, log 40).  Nothing here is a claim about mechanisms or robot capability; see the P0
report for the numerical receipts.
