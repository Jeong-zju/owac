# N02 independent acceptance receipts

The audited base is `dfece06`, following `19ecc65`. These small artifacts support
the repaired engineering prototype and record failed learned-mechanism gates.
They do not establish physical geometry, PI05 ambiguity handling or closed-loop
task benefit. The canonical Chinese audit is
[2026-10-12_n02_encoder.md](../../2026-10-12_n02_encoder.md#独立验收与修补).

- `source-manifest.json`: final source/config/test SHA256 snapshot.
- `validation.json`: local checks, exact skip reasons and evidence boundaries.
- `train-*.json` / `learned-*.json`: independently trained controls and held-out
  per-frame records. Correct, misleading and missing prior interventions reuse
  identical raw bytes. Slot-parity gates do not apply to single-field/PF arms.
- `loss-ablation-*-{config,train,learned}.json` / `loss-ablations.json`: each term
  set to zero in a separate 200-step run; same initialization, seed and data.
- `reproducibility.json`: two final CPU 200-step checkpoints are byte-identical.
- `token-localization-fixture.json`: actual 256x2048 tokens consumed by a fitted
  decoder; explicit incompatible-field fixture, no geometry shortcut or PI05.
- `{libero,robotwin}-real.json`: actual author NPZ, checkpoint/stats/source hashes,
  20 valid batches, encoder/readout gradients, frozen backend audit and 10 policy
  inference timings. MAP readout only; no validation action score.
- `{libero,robotwin}-split.json`: acquisition-source split and selected NPZ hashes.
- `libero-resume.json`: true optimizer/RNG/cursor restore from 20 to 21 steps.
- `remote-smoke-reproduce.sh`: exact runtime/assets with fresh output directories.
  Select `N02_SMOKE_OUTPUT` if its default already exists; old evidence is retained.
- `historical-before-class-loss-*`, `train-plain-unmatched.json`: earlier failed
  or capacity-mismatched results, preserved as history.
- `reference-*`: analytic/reference-only results, never learned-mechanism PASS.

Large checkpoints, datasets and logs remain in ignored local outputs or the
isolated server research directory. The frozen benchmark and services were not
modified by this audit.
