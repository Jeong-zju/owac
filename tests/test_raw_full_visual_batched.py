"""CPU equivalence tests for the vectorized full-visual temporal forward."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from owac.baselines.raw_temporal_siglip import TemporalRawStemStudent
from owac.training.raw_distillation import token_distillation_loss
from owac.training.raw_full_visual_batched import (
    BatchedTemporalError,
    batched_anchor_forward,
)

_HELPER_PATH = Path(__file__).parent / "test_raw_temporal_siglip.py"
_SPEC = importlib.util.spec_from_file_location("batched_temporal_helpers", _HELPER_PATH)
_HELPERS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HELPERS)


def _sample(start: int, *, source: str, present=None, cameras=("base",)):
    base = 0.10 if source == "source-a" else 0.20
    values = tuple(base + 0.01 * index for index in range(6))
    unroll = []
    for offset in range(0, 30, 5):
        endpoint = start + offset
        unroll.append(
            _HELPERS.make_packet(
                endpoint,
                source=source,
                reset=endpoint == 0,
                cameras=cameras,
                present=present,
                values=values,
            )
        )
    prefix = []
    for endpoint in range(0, start, 5):
        prefix.append(
            _HELPERS.make_packet(
                endpoint,
                source=source,
                reset=endpoint == 0,
                cameras=cameras,
                present=present,
                values=values,
            )
        )
    return SimpleNamespace(
        source_id=source,
        endpoint=start + 25,
        prefix_packets=tuple(prefix),
        unroll_packets=tuple(unroll),
    )


def _serial_tokens(student, sample):
    state = None
    with torch.no_grad():
        for packet in sample.prefix_packets:
            state = student.update_state(packet, state)
        if state is not None:
            state = student.detach_state(state)
    for packet in sample.unroll_packets[:-1]:
        state = student.update_state(packet, state)
    output, _ = student(sample.unroll_packets[-1], state)
    return output.projected_tokens, output.tokens.mask


def _serial_loss(student, samples, teachers):
    total = 0.0
    for sample, teacher in zip(samples, teachers, strict=True):
        tokens, mask = _serial_tokens(student, sample)
        total = total + token_distillation_loss(tokens, teacher, mask).total
    return total


def test_batched_tokens_loss_and_all_trainable_gradients_match_serial() -> None:
    samples = [
        _sample(0, source="source-a"),
        _sample(5, source="source-b", present=((True,),)),
        _sample(10, source="source-a"),
    ]
    serial = TemporalRawStemStudent(_HELPERS.fixture_student())
    batched = copy.deepcopy(serial)
    probe = batched_forward_forward(batched, samples)
    token_width = int(probe.projected_tokens.shape[1])
    teachers = [
        torch.randn(1, token_width, 6, generator=torch.Generator().manual_seed(100 + index))
        for index in range(3)
    ]

    serial_loss = _serial_loss(serial, samples, teachers)
    serial.zero_grad(set_to_none=True)
    serial_loss.backward()
    serial_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in serial.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }

    batched.zero_grad(set_to_none=True)
    result = batched_forward_forward(batched, samples)
    batched_loss = 0.0
    for index, teacher in enumerate(teachers):
        batched_loss = (
            batched_loss
            + token_distillation_loss(
                result.projected_tokens[index : index + 1],
                teacher,
                result.token_mask[index : index + 1],
            ).total
        )
    batched_loss.backward()
    batched_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in batched.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    assert torch.allclose(serial_loss, batched_loss, rtol=1e-5, atol=1e-6)
    assert set(serial_grads) == set(batched_grads)
    for name in serial_grads:
        assert torch.allclose(serial_grads[name], batched_grads[name], rtol=1e-5, atol=1e-6), name
    for index, sample in enumerate(samples):
        serial_tokens, serial_mask = _serial_tokens(serial, sample)
        assert torch.allclose(
            serial_tokens,
            result.projected_tokens[index : index + 1],
            rtol=1e-5,
            atol=1e-6,
        )
        assert torch.equal(serial_mask, result.token_mask[index : index + 1])


def batched_forward_forward(student, samples):
    # Module method would dispatch through DDP in the active runtime; this test
    # exercises the vectorized primitive directly.
    return batched_anchor_forward(student, samples)


def test_batched_order_is_permutation_equivariant_and_isolated() -> None:
    samples = [
        _sample(0, source="source-a"),
        _sample(5, source="source-b"),
        _sample(10, source="source-a"),
    ]
    student = TemporalRawStemStudent(_HELPERS.fixture_student())
    forward = batched_forward_forward(student, samples)
    reordered = batched_forward_forward(student, [samples[2], samples[0], samples[1]])
    assert torch.allclose(
        forward.projected_tokens[[2, 0, 1]], reordered.projected_tokens, rtol=1e-5, atol=1e-6
    )
    assert torch.equal(forward.token_mask[[2, 0, 1]], reordered.token_mask)
    assert forward.source_ids == ("source-a", "source-b", "source-a")
    assert reordered.source_ids == ("source-a", "source-a", "source-b")


def test_batched_camera_order_mismatch_is_rejected() -> None:
    samples = [_sample(0, source="source-a"), _sample(5, source="source-b", cameras=("wrist",))]
    student = TemporalRawStemStudent(_HELPERS.fixture_student())
    with pytest.raises(BatchedTemporalError, match="camera order"):
        batched_forward_forward(student, samples)


def test_time_varying_camera_presence_matches_native_burst_masks() -> None:
    import dataclasses

    sample = _sample(5, source="source-a")
    packets = []
    for packet in sample.prefix_packets:
        packets.append(
            dataclasses.replace(
                packet,
                frames=tuple(
                    dataclasses.replace(frame, camera_present=((False,),)) if index < 2 else frame
                    for index, frame in enumerate(packet.frames)
                ),
            )
        )
    sample.prefix_packets = tuple(packets)
    unroll = []
    for packet in sample.unroll_packets:
        unroll.append(
            dataclasses.replace(
                packet,
                frames=tuple(
                    dataclasses.replace(frame, camera_present=((False,),)) if index < 2 else frame
                    for index, frame in enumerate(packet.frames)
                ),
            )
        )
    sample.unroll_packets = tuple(unroll)
    student = TemporalRawStemStudent(_HELPERS.fixture_student())
    batched = copy.deepcopy(student)
    serial_tokens, serial_mask = _serial_tokens(student, sample)
    result = batched_forward_forward(batched, [sample])
    assert torch.allclose(serial_tokens, result.projected_tokens, rtol=1e-5, atol=1e-6)
    assert torch.equal(serial_mask, result.token_mask)


def test_invalid_endpoint_sequence_is_rejected() -> None:
    sample = _sample(5, source="source-a")
    broken = SimpleNamespace(
        source_id=sample.source_id,
        endpoint=sample.endpoint,
        prefix_packets=sample.prefix_packets,
        unroll_packets=(
            sample.unroll_packets[0],
            sample.unroll_packets[2],
            sample.unroll_packets[3],
            sample.unroll_packets[4],
            sample.unroll_packets[5],
            sample.unroll_packets[5],
        ),
    )
    student = TemporalRawStemStudent(_HELPERS.fixture_student())
    with pytest.raises(BatchedTemporalError, match="endpoint sequence"):
        batched_forward_forward(student, [broken])


def _batched_ddp_worker(
    rank: int,
    init_file: str | None = None,
    queue: Any = None,
    output_dir: str | None = None,
) -> None:
    import importlib.util
    from types import SimpleNamespace

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from owac.baselines.raw_temporal_siglip import TemporalRawStemStudent
    from owac.training.raw_distillation import token_distillation_loss
    from owac.training.raw_full_visual_batched import (
        BatchedTemporalRequest,
    )

    spec = importlib.util.spec_from_file_location(
        f"ddp_batch_helpers_{rank}", Path(__file__).parent / "test_raw_temporal_siglip.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helpers)

    def make_sample(start, source):
        base = 0.10 if source == "source-a" else 0.20
        values = tuple(base + 0.01 * index for index in range(6))
        unroll = tuple(
            helpers.make_packet(
                start + offset,
                source=source,
                reset=(start + offset) == 0,
                cameras=("base",),
                values=values,
            )
            for offset in range(0, 30, 5)
        )
        prefix = tuple(
            helpers.make_packet(
                endpoint,
                source=source,
                reset=endpoint == 0,
                cameras=("base",),
                values=values,
            )
            for endpoint in range(0, start, 5)
        )
        return SimpleNamespace(
            source_id=source,
            endpoint=start + 25,
            prefix_packets=prefix,
            unroll_packets=unroll,
        )

    all_samples = [
        make_sample(0, "source-a"),
        make_sample(5, "source-b"),
        make_sample(10, "source-a"),
        make_sample(15, "source-b"),
    ]
    local = all_samples[rank * 2 : rank * 2 + 2]
    teachers = [
        torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(200 + index))
        for index in range(4)
    ]
    counts = [4, 4, 4, 4]
    global_tokens = 16
    world = 2
    if init_file is None:
        dist.init_process_group("gloo")
    else:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
        )
    student = TemporalRawStemStudent(helpers.fixture_student())
    ddp = DistributedDataParallel(student)
    local_loss = 0.0
    # DDP module call: actual hooks fire on this one vectorized batch backward.
    result = ddp(BatchedTemporalRequest(tuple(local)))
    for index in range(len(local)):
        global_index = rank * 2 + index
        term = token_distillation_loss(
            result.projected_tokens[index : index + 1],
            teachers[global_index],
            result.token_mask[index : index + 1],
        ).total
        local_loss = local_loss + term * (world * counts[global_index] / global_tokens)
    local_loss.backward()
    ddp_grad = student.gate_logit.grad.detach().clone()
    ddp_stem = student.stem.conv64.weight.grad.detach().clone()

    # Independent serial reference over all four samples, same weights/teachers.
    reference = TemporalRawStemStudent(helpers.fixture_student())
    reference_loss = 0.0
    for index, sample in enumerate(all_samples):
        state = None
        with torch.no_grad():
            for packet in sample.prefix_packets:
                state = reference.update_state(packet, state)
            if state is not None:
                state = reference.detach_state(state)
        for packet in sample.unroll_packets[:-1]:
            state = reference.update_state(packet, state)
        final, _ = reference(sample.unroll_packets[-1], state)
        term = token_distillation_loss(
            final.projected_tokens,
            teachers[index],
            final.tokens.mask,
        ).total
        reference_loss = reference_loss + term * (counts[index] / global_tokens)
    reference_loss.backward()
    ref_gate = reference.gate_logit.grad.detach().clone()
    ref_stem = reference.stem.conv64.weight.grad.detach().clone()
    payload = {
        "rank": rank,
        "ddp_gate": ddp_grad.tolist(),
        "ddp_stem": ddp_stem.tolist(),
        "ref_gate": ref_gate.tolist(),
        "ref_stem": ref_stem.tolist(),
    }
    if queue is not None:
        queue.put(payload)
    if output_dir is not None:
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(payload))
    dist.destroy_process_group()


def test_two_rank_gloo_batched_weighted_gradient_matches_serial_global(tmp_path) -> None:
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    output = tmp_path / "ddp-results"
    output.mkdir()
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--master_port",
        str(port),
        str(Path(__file__)),
        "--worker",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    outputs = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert {item["rank"] for item in outputs} == {0, 1}
    for item in outputs:
        assert torch.allclose(
            torch.tensor(item["ddp_gate"]), torch.tensor(item["ref_gate"]), rtol=1e-6, atol=1e-7
        )
        assert torch.allclose(
            torch.tensor(item["ddp_stem"]), torch.tensor(item["ref_stem"]), rtol=1e-6, atol=1e-7
        )


if __name__ == "__main__" and "--worker" in sys.argv:
    _batched_ddp_worker(
        int(os.environ["RANK"]),
        output_dir=sys.argv[sys.argv.index("--worker") + 1],
    )
