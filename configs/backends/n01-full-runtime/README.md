# N01 v1.0 independent GPU runtime (remote)

P0 establishes a remote, independent environment under
`/workspace/owac/rawvla/research/n01-v1/`. Nothing in this directory is installed
locally; the local worktree only produces and verifies the source bundle.

## Local vs remote locks are separate

| | local worktree | remote GPU environment |
| --- | --- | --- |
| lock | `uv.lock` + `pyproject.toml` encoder extra | `requirements-gpu.txt` (this directory) |
| torch | `2.9.1` **CPU** wheel from the `pytorch-cpu` index | CUDA wheel (`torch==2.9.1+cu128` verified) |
| purpose | code, FP64 numerical reference, regression tests | profiling, motion adaptation, training, replay |
| python | system/`.venv`, project requires `>=3.11` | encoder env `>=3.11`; the frozen policy env stays 3.10.21 |

The local CPU lock must never be installed into the remote environment and the
remote CUDA lock must never be installed into the local `.venv`. `uv run --locked
--extra encoder` always means the CPU lock.

## Layout to create

```text
/workspace/owac/rawvla/research/n01-v1/
  # encoder is at /workspace/owac/rawvla/envs/encoder (existing verified venv)
  envs/policy/                  # separate policy env when new deps are needed
  runs/<run_id>/source/         # unpacked source bundle (this stage's snapshot)
  runs/<run_id>/outputs/        # logs, curves, checkpoints, per-event records
  runs/<run_id>/receipts/       # small provenance, cost and validation records
```

## Build the encoder environment

```bash
# 1. On the server, create the independent env (do not touch the StarVLA env).
uv venv --python 3.11 /workspace/owac/rawvla/envs/encoder

# uv-created environment does not include pip; use uv pip --python.

# 2. Install the GPU lock from the unpacked source snapshot.
cd /workspace/owac/rawvla/research/n01-v1/runs/<run_id>/source
uv pip install --python /workspace/owac/rawvla/envs/encoder/bin/python \
    -r configs/backends/n01-full-runtime/requirements-gpu.txt

# 3. Freeze what was actually installed and keep it with the run receipts.
uv pip freeze --python /workspace/owac/rawvla/envs/encoder/bin/python \
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
PYTHONPATH="$RUN/source/src" /workspace/owac/rawvla/envs/encoder/bin/python \
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
5. `budget.pending_profiling` now contains P0 measurements and explicitly scaled
   estimates. The factory profiles only existing four-channel DBSR modules; the
   three-channel encoder, SEA-RAFT and support replay remain later-stage work.

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

## Reproduce the accepted P0 entry

```bash
PYTHONPATH=src /workspace/owac/rawvla/envs/encoder/bin/python -m owac.runtime.n01_full_preflight profile --mode collect --config configs/encoders/n01_full/protocol_v1.json --output <new-output-directory>
```

The factory is `owac.runtime.n01_full_profile:dbsr_factory`. It measures full
64/9/512 backbone/fusion modules and labels `entry=direct` above eight frames.
It is not the P4 stateful encoder. Cost estimates use the declared 128/8 point,
8 gradient microsteps, all seven arms, three seeds, and 2/3 platform cameras.
Use `owac.raw.n01_full_manifest.dataset_from_protocol` to construct each real
loader; this binds partition and statistics selection to the same protocol.
