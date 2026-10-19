"""GPU 路径合同：设备搬运、AMP 前反向、开发集评测与缓存导出。

这些测试在本地 RTX 5090 与远端 RTX PRO 6000 上跑同一段代码；CPU 版本不覆盖
``autocast`` 与 ``.to(device)`` 路径，而这两处正是 S1 首轮 smoke 暴露的缺陷。
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from owac.training.n04_full import (
    SequenceStore,
    Trainer,
    build_index,
    build_scene_views,
    export_main,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def store(s1_config) -> SequenceStore:
    return SequenceStore(max_frames_per_sequence=s1_config.data.max_frames_per_sequence)


@pytest.fixture(scope="module")
def cuda_available() -> bool:
    if not torch.cuda.is_available():
        pytest.skip("需要可用的 CUDA 设备")
    return True


def _gpu_config(s1_config, cuda_available):
    training = replace(
        s1_config.training, batch_frames=4, micro_batch_frames=4, amp=True, device="cuda"
    )
    return replace(s1_config, training=training)


def test_cuda_measure_uses_model_device(s1_config, store, cuda_available):
    config = _gpu_config(s1_config, cuda_available)
    trainer = Trainer(config, seed=17, run_dir="/tmp/n04_s1_gpu_measure")
    groups = build_index(config, ("train",), store)
    ref = groups[("libero", (64, 48))][0]
    frame, _, _, _ = build_scene_views(ref, store)
    with torch.no_grad():
        out = trainer.model.measure(
            [frame.move_to(device=trainer.device)], autocast_dtype=torch.bfloat16
        )
    assert out.depth_mean.device.type == "cuda"
    assert bool(torch.isfinite(out.depth_mean).all())
    # 结果与 CPU 前向在 bf16 下的差异必须有限且量级合理
    cpu_model = Trainer(config, seed=17, run_dir="/tmp/n04_s1_gpu_measure_cpu")
    cpu_model.model = cpu_model.model.to("cpu")
    cpu_model.device = torch.device("cpu")
    with torch.no_grad():
        cpu_out = cpu_model.model.measure([frame])
    relative = float(
        (out.depth_mean.cpu() - cpu_out.depth_mean).abs().mean() / cpu_out.depth_mean.mean()
    )
    # 未训练网络上 bf16/autocast 与 fp32 的差异只做量级 sanity 检查；精度契约属于 S4
    assert relative < 0.15


def test_cuda_training_step_and_evaluate(s1_config, store, cuda_available, tmp_path):
    config = _gpu_config(s1_config, cuda_available)
    trainer = Trainer(config, seed=17, run_dir=tmp_path / "gpu")
    trainer.train_groups = build_index(config, ("train",), store)
    info = trainer.train_step(0)
    assert all(torch.isfinite(torch.tensor(value)) for key, value in info.items() if key != "count")
    for name, parameter in trainer.model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    metrics = trainer.evaluate("dev")
    assert metrics["depth_mae_m"] is not None
    assert metrics["pixels"] > 0


def test_cuda_export_matches_online(s1_config, store, cuda_available, tmp_path):
    config = _gpu_config(s1_config, cuda_available)
    trainer = Trainer(config, seed=17, run_dir=tmp_path / "gpu_export")
    checkpoint = tmp_path / "gpu_export" / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    report = export_main(
        config, tmp_path / "gpu_cache", str(checkpoint), splits=("dev",), verify_samples=2
    )
    assert report["entries"] > 0
    for result in report["verification"]["results"]:
        assert result["entries_match"] is True
        assert (
            max(
                result["abs_max_diff"].values()
                if "abs_max_diff" in result
                else result["max_abs_diff"].values()
            )
            < 1e-5
        )
