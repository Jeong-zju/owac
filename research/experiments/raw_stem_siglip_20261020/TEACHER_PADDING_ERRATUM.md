# Teacher padding erratum — 2026-10-20

Sol's round12 alarm was based on the PI05 class default and the YAML's
`tokenizer.pad_state_value: -2.0` without checking the effective runtime flag.

The actual LIBERO config explicitly sets:

```yaml
framework:
  discrete_state_input: false
tokenizer:
  pad_state_value: -2.0
```

`PI0._init_runtime_config` overwrites the instance flag from that config. With
`discrete_state_input=false`:

- `_prepare_examples` pads state to 32 with **0.0**;
- `_build_observation_from_batch` passes **None** as tokenizer state;
- the tokenizer therefore does not digitize the state vector at all.

The historical actual checkpoint config also has `discrete_state_input: false`
and `pad_state_value: 0.0`. The existing `owac.backends.rawvla.new_model`
LIBERO path explicitly sets false and pad0.

Therefore the original explicit zero-padded32 was **bitwise equivalent** to
native state8→0 padding for this LIBERO mode. The physical 155 updates were not
invalidated by this padding issue. Sol's alarm is disproven by the effective
config and actual tokenizer behavior.

The round12 interface cleanup (model-facing state8) is retained as a generic API
improvement and preserves the false-mode output. Tests now drive the pinned
`_prepare_examples`/`_build_observation_from_batch` implementation and prove:

- false mode + configurable -2 attribute still pads 0 and does not tokenize state;
- state8 and old zero32 produce identical state tensors/images/masks/observations
  and identical tokenizer behavior;
- true mode uses its actual configured pad and does tokenize state.

Evidence:

- `outputs/rawvla-teacher-20261020/round-13/effective-config.json`
- `outputs/rawvla-teacher-20261020/round-13/effective-config-and-tokenizer-proof.json`
- `tests/test_rawvla_teacher_state_padding.py`

No optimizer, counters, old receipts, metrics, or checkpoints were changed.
The 9th interrupted validation episode is retained as an operator-error invalid
attempt in the full denominator; the 8 completed successes remain valid under
the actual false config. No 10/10 or 20-state acceptance is claimed.
