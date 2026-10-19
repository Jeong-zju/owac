"""Scientific plots of same-observation diagnosis, not closed-loop scores."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    checkout = Path(__file__).resolve().parents[4]
    root = checkout / "outputs/n02-v12/visual-gap-20261016"
    reports = {
        m: json.loads((root / "a1" / m / "report.json").read_text()) for m in ("native", "raw-warm")
    }
    contrib = json.loads((root / "level_contributions.json").read_text())
    plt.rcParams.update(
        {"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "font.size": 11}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    colors = {"native": "#287b9b", "raw-warm": "#c8583b"}
    names = {"native": "原始 π0.5", "raw-warm": "当前 RAW r7"}
    for mode, d in reports.items():
        values = [
            d["counterfactuals"][f"noise_{offset}"]["first_target_delta_max_rad"] * 180 / np.pi
            for offset in (50, 100, 150, 200)
        ]
        axes[0].plot(range(1, 5), values, "o-", label=names[mode], color=colors[mode], lw=2)
    axes[0].set(
        title="画面和状态不变，只换采样噪声",
        xlabel="相对于同一基准的四个噪声种子",
        ylabel="首条关节目标的最大变化（度）",
        xticks=range(1, 5),
        ylim=(0, 100),
    )
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    values = contrib["sampled_level_rms"]
    axes[1].bar(
        ["浅层 stride4", "中层 stride8", "深层 stride16"],
        values,
        color=["#75b0a0", "#47948e", "#c8583b"],
        width=0.6,
    )
    axes[1].set(
        title="RAW 多尺度特征拼接前的幅度", ylabel="RMS（对数刻度）", yscale="log", ylim=(0.1, 100)
    )
    for i, v in enumerate(values):
        axes[1].text(i, v * 1.18, f"{v:.3f}", ha="center", fontsize=12)
    axes[1].text(
        0.02,
        0.95,
        "深层占拼接输入平方能量的 99.97%",
        transform=axes[1].transAxes,
        va="top",
        color="#a33b29",
    )
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("同一真实初态的离线诊断 · 固定语言与状态 · 优化更新 0", fontsize=14)
    fig.savefig(Path(__file__).with_name("diagnosis.png"), dpi=160)


if __name__ == "__main__":
    main()
