# N01 v1.0 independent GPU runtime (remote)

P0 establishes a remote, independent environment under
`/workspace/owac/rawvla/research/n01-v1/`. Nothing in this directory is installed
locally; the local worktree only produces and verifies the source bundle.

## Local vs remote locks are separate

| | local worktree | remote GPU environment |
| --- | --- | --- |
| lock | `uv.lock` + `pyproject.toml` encoder extra | `requirements-gpu.txt` (this directory) |
| torch | `2.9.1` **CPU** wheel from the `pytorch-cpu` index | CUDA wheel (`torch==2.7.1+cu128` candidate) |
| purpose | code, FP64 numerical reference, regression tests | profiling, motion adaptation, training, replay |
| python | system/`.venv`, project requires `>=3.11` | encoder env `>=3.11`; the frozen policy env stays 3.10.21 |

The local CPU lock must never be installed into the remote environment and the
remote CUDA lock must never be installed into the local `.venv`. `uv run --locked
--extra encoder` always means the CPU lock.

## Layout to create

```text
/workspace/owac/rawvla/research/n01-v1/
  envs/encoder/                 # independent GPU encoder / SEA-RAFT / Mamba env
  envs/policy/                  # separate policy env when new deps are needed
  runs/<run_id>/source/         # unpacked source bundle (this stage's snapshot)
  runs/<run_id>/outputs/        # logs, curves, checkpoints, per-event records
  runs/<run_id>/receipts/       # small provenance, cost and validation records
```

## Build the encoder environment

```bash
# 1. On the server, create the independent env (do not touch the StarVLA env).
python3.11 -m venv /workspace/owac/rawvla/research/n01-v1/envs/encoder
/workspace/owac/rawvla/research/n01-v1/envs/encoder/bin/python -m pip install -U pip

# 2. Install the GPU lock from the unpacked source snapshot.
cd /workspace/owac/rawvla/research/n01-v1/runs/<run_id>/source
/workspace/owac/rawvla/research/n01-v1/envs/encoder/bin/pip install \
    -r configs/backends/n01-full-runtime/requirements-gpu.txt

# 3. Freeze what was actually installed and keep it with the run receipts.
/workspace/owac/rawvla/research/n01-v1/envs/encoder/bin/pip freeze \
    > receipts/encoder-pip-freeze-$(date -u +%Y%m%dT%H%M%SZ).txt
```

If the machine needs a proxy or an internal mirror, configure it outside the
repository and record it in the run receipt; never commit credentials.

## Source snapshot and hash verification

The bundle produced by `owac.runtime.n01_full_preflight bundle` contains
`source.tar.gz` plus `bundle-manifest.json`. On the server:

```bash
RUN=/workspace/owac/rawvla/research/n01-v1/runs/<run_id>
mkdir -p "$RUN/source" "$RUN/outputs" "$RUN/receipts"
tar -xzf source.tar.gz -C "$RUN/source"
# Re-run the same verify command on the unpacked copy:
PYTHONPATH="$RUN/source/src" /workspace/owac/rawvla/research/n01-v1/envs/encoder/bin/python \
    -m owac.runtime.n01_full_preflight verify --bundle <unpacked-bundle-dir> \
    --source-root "$RUN/source" --report "$RUN/receipts/verify.json"
```

The bundle is built by walking the filesystem, so new untracked sources are
included; `git archive HEAD` is never used.

## Acceptance before anything is marked `ready`

1. `torch.cuda.is_available()` is true and `torch.cuda.get_device_capability()`
   matches the Blackwell device;
2. a real forward+backward on the configured shapes succeeds
   (`owac.runtime.n01_full_preflight profile --mode collect` refuses to run
   anywhere without CUDA, so this is the same entry point as the profiling);
3. the local and remote `bundle-manifest.json` file hashes agree;
4. the memory budget re-measures free device memory and counts the still-running
   services; other projects' services are never stopped;
5. unresolved until measured: the actual step time, valid batch, max steps and
   GPU-hour estimate stay `pending_profiling` in `protocol_v1.json`.

## Supervisor

`owac-n01-v1.supervisor.conf` is the template. Every long-running job is one
supervisor program named `owac-n01-v1-<stage>-<run>`; the wrapper calls the
explicit Python, source root and config. Logs go to both the run `outputs/` and
the supervisor log file. Register per the server guide, keep
`autostart=false`, and never let a resumed job silently restart from step 0.

## Later stages

P5 adds Mamba and P6 adds BSRT. Their kernels/backbones are locked as source and
dependency availability only; build and record them in this environment (or a
new one) with the compile flags written into the run receipt. The policy
environment stays Python 3.10-compatible and loads OWAC through
`PYTHONPATH=<snapshot>/src:<author sources>/starVLA`.
