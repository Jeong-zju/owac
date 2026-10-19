"""Assemble one complete, copyable DeepSeek prompt per N01 v1.2 stage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED = (
    "00_SHARED_CONTEXT.md",
    "04_RAW_RGB_PROTOCOL.md",
    "01_REFERENCE_CODE.md",
    "02_REMOTE_EXECUTION.md",
    "03_FRONTEND_EXPERT.md",
)
STAGES = (
    ("P3", "FULL_ENCODER", "完整编码器 真实训练与开发闭环"),
    ("P4", "FROZEN_FULL_EVALUATION", "冻结后的完整基准验证"),
    ("P5", "TRANSFER_AND_EXTENSIONS", "原生RAW 跨域与有依据的扩展"),
)


def main():
    shared = "\n\n".join((ROOT / name).read_text().strip() for name in SHARED)
    for stage, name, title in STAGES:
        body = (ROOT / "stages" / f"{stage}.md").read_text().strip()
        packet = (
            f"# DeepSeek N01 v1.2 {stage} {title}\n\n"
            f"本次仅执行 **N01 v1.2 {stage}**。这是一份完整交接包，"
            "共同要求、参考实现、远程训练、前端专家任务和阶段指令均已展开；"
            "请从头读完再执行，不需要用户手动拼接其他提示词。"
            "准备本包时P3尚未实施，后续阶段的前置交付必须以届时实际记录为准。\n\n"
            f"{shared}\n\n{body}\n"
        )
        target = ROOT / f"{stage}_{name}.md"
        target.write_text(packet)
        print(f"{target.name}: {len(packet)} characters")


if __name__ == "__main__":
    main()
