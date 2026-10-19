# N02 full runtime (P0 §2.3)

> Current independent acceptance: [tested reference environments and compiler patch](INDEPENDENT_ACCEPTANCE.md). The old Theseus/SAVi/Minkowski limitations below describe the first pass and are superseded where that record provides actual execution evidence.

An isolated **Python 3.10 + CUDA 12.8** environment for the N02 full-development line. It exists
because the root `owac` package requires Python >=3.11 and its lock pins CPU torch, while the
already deployed RawVLA / StarVLA π0.5 backend runs on Python 3.10 with a CUDA stack. The
integration target for P0 is therefore "Python 3.10 + GPU stack", and this directory defines it.

Everything below is measured, not assumed: the receipts live in
`outputs/n02-full/p0/runtime/` (probe JSON) and `outputs/n02-full/p0/runtime/logs/`.

## Purpose

- Run real CUDA kernels on the local RTX 5090 (`sm_120` needs CUDA >= 12.8 wheels).
- Host the upstream component reproduction environments (Theseus, MinkowskiEngine, JAX/Gen
  references get their own isolated envs; see `../../docs/status.md`).
- Load the project code of this checkout through `PYTHONPATH=src`. The root package is **not**
  installed here, so its `requires-python = ">=3.11"` never constrains this runtime, and this
  runtime never constrains the root lock.
- The Python 3.10 syntax/standard-library barrier of `src/owac` is checked with
  `python -m compileall -q src/owac` under this interpreter (exit 0 on 2026-10-14).

## Actual resolved versions (measured 2026-10-14)

| Item | Value | Receipt |
| --- | --- | --- |
| Interpreter | `/home/jeong/zeno/owac/worktrees/n02-measurement-completion/outputs/n02-full/runtime/py310-cu128/bin/python` | `probe.json` |
| Python | **3.10.19** (uv-managed `cpython-3.10.19-linux-x86_64-gnu`) | `probe.json`, `logs/01_uv_venv_py310.log` |
| torch | **2.11.0+cu128** | `probe.json`, `logs/03_uv_pip_install_retry.log` |
| torchvision | **0.26.0+cu128** | `probe.json` |
| numpy | 2.2.6 | `requirements.lock.txt` |
| `torch.version.cuda` | 12.8 | `probe.json` |
| cuDNN | 9.19.0 (`torch.backends.cudnn.version() == 91900`) | `probe.json` |
| GPU | NVIDIA GeForce RTX 5090, capability `[12, 0]` (`sm_120`) | `probe.json` |
| VRAM | 32,088.5 MiB total by torch / 32,607 MiB by `nvidia-smi`; free at probe time recorded per run | `probe.json`, `resources.json` |
| Driver | 595.99.02 (driver-reported CUDA 13.2) | `resources.json` |
| torch stack accuracy vs FP64 (CPU reference) | matmul fp32 2.12472e-05 / bf16 0.224387; conv2d fp32 5.72251e-06 / bf16 0.148582 (max abs error) | `probe.json` |

`torch==2.11.0+cu128` is the newest CUDA 12.8 wheel that still publishes a cp310 build; it is what
uv resolved, and it is written here as measured. There is no torch 3.x in this environment.

## Create / recreate (reproducible)

```bash
cd /home/jeong/zeno/owac/worktrees/n02-measurement-completion

# 1. exact procedure that produced the environment in use
uv venv --python 3.10 outputs/n02-full/runtime/py310-cu128
uv pip install --python outputs/n02-full/runtime/py310-cu128/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 2. byte-identical rebuild from the lock file (verified 2026-10-14, logs/13)
uv venv --python 3.10 outputs/n02-full/runtime/verify-repro
uv pip install --python outputs/n02-full/runtime/verify-repro/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  -r configs/backends/n02-full-runtime/requirements.lock.txt
uv pip freeze --python outputs/n02-full/runtime/verify-repro/bin/python   # identical to the lock
```

`uv venv` prints a warning that the requested interpreter is below the root project's
`requires-python`; that warning is expected and harmless here (the runtime must be 3.10). Passing
`--no-project` suppresses it and was used for the isolated Theseus env (`logs/14`).

Run the probe (also the required P0 self-check):

```bash
PYTHONPATH=src outputs/n02-full/runtime/py310-cu128/bin/python \
  -m owac.runtime.n02_full_probe --device cuda \
  --output outputs/n02-full/p0/runtime/probe.json
```

## Relationship to the deployed StarVLA Python 3.10 target

- The rented RawVLA server runs Python 3.10 with **torch 2.7.1+cu128** on an RTX PRO 6000
  Blackwell (97,887 MiB), per `research/experiments/2026-10-10_rawvla_pi05_deployment.md`. That
  host was **not reachable** during this task, so its current state is unverified.
- This local runtime matches the *Python minor version* (3.10.19) of that target, but resolves a
  newer torch (2.11.0+cu128 vs 2.7.1+cu128). Both are cu128 builds; a component that only works
  on 2.7.1 must be re-verified on the server, not assumed here.
- Because the environment is 3.10, `PYTHONPATH=src` imports of `owac` modules execute under the
  same interpreter as CUDA kernels: no cross-environment process hop is needed for the
  differentiable path (this is a property of the interpreter, not a claim that gradients cross
  process boundaries).

## Known incompatibilities and gaps

- Root `owac` package (`requires-python >=3.11`) is not installable here; that is the reason the
  runtime uses `PYTHONPATH=src`.
- `pytest` **is** installed here (9.1.1) so the P0 test suite can be executed by the target
  interpreter itself; it is part of `requirements.lock.txt`. The N02 probe unit tests also run in
  the root `.venv` (CPU) and on the CUDA conda interpreters.
- CUDA toolkit 12.8 is present locally at `/usr/local/cuda-12.8` (`nvcc` 12.8.61), but the bare
  `nvcc` on `PATH` is `/usr/bin/nvcc` = CUDA **12.0**, so any CUDA extension build must set
  `CUDA_HOME=/usr/local/cuda-12.8` explicitly. cuDNN headers are not under `/usr/local/cuda-12.8`
  but are provided by this environment's `nvidia-cudnn-cu12` wheel.
- Theseus: `theseus-ai==0.2.3` resolves here (cp310 is the newest wheel tag), but its shipped
  extensions request the **CUDA 11** sonames (`libcublas.so.11`, `libcudart.so.11.0`,
  `libcusolver.so.11`, `libcusparse.so.11`) and fail to load against this CUDA 12.8 stack
  (`ImportError: libcublas.so.11: cannot open shared object file`). A full install is additionally
  blocked by `scikit-sparse` needing SuiteSparse headers (`cholmod.h` is absent locally). Theseus
  therefore needs its own source build or an isolated CUDA 11 reference env — see
  `outputs/n02-full/p0/runtime/runtime_matrix.json` (`upstream_components.theseus_ai`).
- MinkowskiEngine: sdist only on PyPI (no wheels of any Python tag); a plain install fails inside
  the isolated build environment (`ImportError: Pytorch not found`) and a real build needs
  `--no-build-isolation` with `CUDA_HOME=/usr/local/cuda-12.8`. Not attempted in P0.
- Package index reachability: `pypi.org` and `download.pytorch.org` both answered HTTP 200 on
  2026-10-14. The first install attempt nevertheless **stalled** mid-download (uv held an
  established TLS socket with 0 bytes read over 8 s); the job was killed and re-run, completing
  from the warm uv cache — see `logs/02_*` (stalled attempt) and `logs/03_*` (successful retry).

## P0 verification executed with this interpreter (2026-10-14)

| Check | Command (from the checkout root) | Result | Log |
| --- | --- | --- | --- |
| Python 3.10 syntax/stdlib | `<py310> -m compileall -q src/owac` | exit 0 | `logs/20_ruff_and_compileall.log` |
| CUDA probe | `PYTHONPATH=src <py310> -m owac.runtime.n02_full_probe --device cuda --output outputs/n02-full/p0/runtime/probe.json` | exit 0, all four operators measured | `logs/11_probe_cuda_py310_runtime.log` |
| P0 CLI help | `PYTHONPATH=src <py310> -m owac.evaluation.n02_full.p0 --help` | exit 0, lists `inventory/parity/profile/report` | `logs/15_p0_cli_help_py310.log` |
| P0 CLI inventory | `PYTHONPATH=src <py310> -m owac.evaluation.n02_full.p0 inventory --output outputs/n02-full/p0/runtime/inventory-py310` | exit 0, `status=not_run` (`config data.cache_root is not configured`), `inventory.json` written | `logs/16_p0_inventory_py310.log` |
| P0 test suite | `PYTHONPATH=src <py310> -m pytest tests/encoders/n02_full -q` | **189 passed, 1 failed, 0 skipped** | `logs/18_py310_p0_tests.log` |

The single failure is `tests/encoders/n02_full/test_p0_cli.py::test_profile_measures_real_updates_and_separates_three_columns`:
`p0.py::_runtime_info` takes the `torch.cuda.is_available()` branch whenever the interpreter has
CUDA, then calls `torch.cuda.get_device_name("cpu")` for a CPU-configured profile, which raises
`ValueError: Expected a cuda device, but got: cpu` and leaves `device_name = None` while the test
expects `"cpu"`. This is **not** a Python 3.10 defect: the same test fails under the py3.11 CUDA
conda interpreter (`/home/jeong/miniconda3/envs/wam-py311`) and passes under the CPU-only root
`.venv`; the trigger is "CUDA-capable torch + `device: cpu` in the config". The owning module is
`src/owac/evaluation/n02_full/p0.py` (not changed by this task).

## Files

| File | Role |
| --- | --- |
| `pyproject.toml` | Declarative definition of the runtime (deps, explicit cu128 index, Python range). For review and re-resolution; it is not what produced `uv.lock`. |
| `requirements.lock.txt` | Authoritative pin set: the real `uv pip freeze` of `outputs/n02-full/runtime/py310-cu128` (39 packages incl. all `nvidia-*-cu12` runtime libs and `pytest`). |
| `README.md` | This file. |

The environment directories themselves live under `outputs/n02-full/runtime/` (gitignored); no
virtual environment is ever created inside `configs/`.
