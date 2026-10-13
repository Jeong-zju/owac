# Third-party code notices

## Deep Burst Super-Resolution

- Authors: Goutam Bhat, Martin Danelljan, Luc Van Gool, Radu Timofte.
- Work: *Deep Burst Super-Resolution*, CVPR 2021.
- Source: https://github.com/goutamgmb/deep-burst-sr
- Pinned revision: `16f158e0553b6702cc1693c43f733c2830973278`.
- License: [CC-BY-NC-SA-4.0 and upstream academic-research-use statement](src/owac/baselines/_vendor/dbsr/LICENSE).

The files under `src/owac/baselines/_vendor/dbsr/` originate from this project. Original copyright and license headers are retained. Changes are limited to relative/unused imports, formatting and docstring whitespace; executable definition AST hashes are checked against the pinned originals. Original paths and SHA-256 hashes are recorded in `provenance.json` in that directory. The license and provenance are included as package data.

`MaskedDbsrFusion` in `src/owac/baselines/burst_features.py` is an OWAC adaptation of upstream `WeightedSum`, adding validity masks, masked consensus and missing-reference handling. This derivative retains the upstream license and academic-use restriction. The OWAC wrapper calls the original feature layers, omitting the RGB/PWC alignment and RGB reconstruction execution paths. It does not include upstream pretrained weights or claim endorsement by the original authors.

`src/owac/frameworks/n01_dbsr.py` executes those same upstream layers with graph-constrained input taps and adapts the fusion forward to this transport rule. This derivative also retains the upstream license and academic-use restriction; no additional trainable layers are supplied.
