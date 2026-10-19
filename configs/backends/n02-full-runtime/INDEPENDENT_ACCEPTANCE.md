# P0 reference environments after independent repair

Repository root for every command:
`/home/jeong/zeno/owac/worktrees/n02-measurement-completion`.
These are isolated local environments. The root package is not installed into Python3.10;
use `PYTHONPATH=src`. No RGB weights are downloaded.

| Purpose | Absolute interpreter (append to repository root) | Lock / limitation |
| --- | --- | --- |
| CUDA component training/profile | `outputs/n02-full/runtime/py310-cu128/bin/python` | `requirements.lock.txt`; Python3.10.19 / torch2.11.0+cu128 / sm_120 |
| SAVi JAX and all-fixture CPU checker | `outputs/n02-full/p0/parity-venv/bin/python` | `requirements.savi-reference.lock.txt`; Python3.12.12 / JAX0.4.38 / Flax0.10.2 |
| Official Theseus dense LM and custom Minkowski CUDA reference | `outputs/n02-full/runtime/py310-theseus/bin/python` | `requirements.theseus-reference.lock.txt`; use pinned Theseus source, not installed wheel source |
| Gen reference | `outputs/n02-full/runtime/julia-1.10.10/bin/julia` | `gen-reference/{Project,Manifest}.toml`; pinned local Gen source |

SAVi and CUDA profile environments pass `uv pip check`. The Theseus environment intentionally
lacks `scikit-sparse`/CHOLMOD development headers; optional wheel sparse-CUDA extensions require
CUDA11 and are not used. The actual official dense example, FP64 LM solution and gradient run on
this Python3.10 / CUDA12.8 torch stack. This does not approve a future sparse GPU solver.

Recreate Python environments with `uv venv --python 3.10` (or 3.12 for SAVi), then install the
corresponding frozen requirements; torch/torchvision `+cu128` and `+cpu` wheels require their
respective PyTorch indexes. For the Theseus environment use `uv pip install --no-deps -r ...`;
its explicit frozen dependency set supports the tested dense path. Do not claim the optional
sparse solvers pass simply because dense LM passes.

For Julia, download the fixed `julia-1.10.10-linux-x86_64.tar.gz` from
`https://julialang-s3.julialang.org/bin/linux/x64/1.10/`, extract under the ignored runtime directory,
and use a private `JULIA_DEPOT_PATH=$PWD/outputs/n02-full/runtime/julia-depot`.
`julia --project=configs/backends/n02-full-runtime/gen-reference -e 'using Pkg; Pkg.instantiate()'`
resolves the frozen manifest against the already pinned `third_party/pinned/probcomp_Gen.jl@2a27298b39`.

## Custom Minkowski reference

Use the actual ConvNeXt-V2 gitlink `shwoo93/MinkowskiEngine@bbc30ef581ea6deb505976b663f5fc2358a83749`.
The generic PyPI engine does not contain the required custom depthwise operator. CPU build can
succeed but that operator explicitly rejects CPU execution.

Extract the fixed codeload tarball into `third_party/pinned/shwoo93_MinkowskiEngine@bbc30ef581`.
Apply `minkowski-cuda128-ownership.patch` once with `patch -p1`. The patch adds missing modern
headers, preserves custom deleter ownership while avoiding an ambiguous C++ std overload, and
moves Python `Sequence` imports to `collections.abc`. It does not change convolution/normalization
formulas. The source lock records patch and patched-file hashes; the original tarball is retained.

From that source directory, create `MinkowskiEngineBackend/` before invoking setup, then build:

```bash
mkdir -p MinkowskiEngineBackend
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 MAX_JOBS=2
export NVCC_PREPEND_FLAGS='-DCCCL_DISABLE_NVTX -include thrust/execution_policy.h -include thrust/remove.h -include thrust/unique.h'
/home/jeong/zeno/owac/worktrees/n02-measurement-completion/outputs/n02-full/runtime/py310-theseus/bin/python setup.py build_ext --inplace --force_cuda --blas=blas
```

The measured build compiled and linked successfully, then its final copy failed because the
backend directory had not been created. Creating the directory and copying the just-built `.so`
completed installation; the actual CUDA parity run then passed. Build attempts 1–7 are retained
under `outputs/n02-full/p0-independent-audit/`; never interpret the compile-only logs as numerical
acceptance. The compiled `.so`, target engine import path and forward/gradient evidence are
identified by the final source/run manifests.

Exact reference exports, parity, real GPU updates and remote data commands are in
`research/experiments/results/2026-10-14_n02_p0_independent_acceptance/reproduce.sh`.
