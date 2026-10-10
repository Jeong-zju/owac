"""Archive Arena numbers only and render a local offline progress page."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from owac.runtime.arena_droid import sha256, write_json

ROOT = "/workspace/owac/reference/pi05-arena-droid-v1"


def completed_episode(seed_dir, seed):
    """Require a complete loop and exactly one boolean official outcome for this seed."""
    summary_path = seed_dir / "evidence/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    integrity_path = seed_dir / "evidence/input_integrity.json"
    if integrity_path.exists() and json.loads(integrity_path.read_text()).get("valid") is False:
        return None, {**summary, "status": "infrastructure_failed", "input_integrity": False}
    records = []
    for path in sorted((seed_dir / "official").rglob("episode_results*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if summary.get("status") != "complete":
        return None, summary
    if len(records) != 1 or type(records[0].get("success")) is not bool:
        raise ValueError(f"Expected exactly one explicit official outcome for seed {seed}")
    episode = records[0]
    if episode.get("seed") != seed or summary.get("completed_episode_count") != 1:
        raise ValueError("Episode seed or completion count mismatch")
    return episode, summary


def archive(args):
    # Explicit subtree allowlist excludes weights, images and RAW binary buffers.
    remote = """import json, pathlib, datetime, hashlib
root = pathlib.Path("/workspace/owac/reference/pi05-arena-droid-v1")
files = {}
for subtree in ("deployment", "runs"):
    for path in sorted((root / subtree).rglob("*")):
        allowed = path.suffix in {".json", ".jsonl", ".yaml", ".txt"}
        if ("raw" not in path.parts and path.is_file() and allowed
                and path.stat().st_size <= 16_000_000):
            files[str(path.relative_to(root))] = path.read_text()
    for path in sorted((root / subtree).rglob("*.log")):
        if path.is_file():
            lines = path.read_text(errors="replace").splitlines()
            files[str(path.relative_to(root)) + ".tail.txt"] = "\\n".join(lines[-100:]) + "\\n"
files["protocol.json"] = (root / "protocol.json").read_text()
verification = root / "verification.json"
if verification.exists():
    files["verification.json"] = verification.read_text()
for name in ("verification-supervisor.log", "raw-supervisor.log", "rgb-supervisor.log"):
    path = root / name
    if path.exists():
        files[name + ".tail.txt"] = "\\n".join(path.read_text(errors="replace").splitlines()[-100:])
sources = {}
for source in sorted(root.glob("owac-src*")):
    sources[source.name] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(source.rglob("*.py"))}
sources["deployment_verifiers"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root / "deployment").glob("verify-*.py"))}
files["source_hashes.json"] = json.dumps(sources, indent=2)
checked = datetime.datetime.now(datetime.UTC).isoformat()
print(json.dumps({"checked_at_utc": checked, "files": files}))
"""
    result = subprocess.run(
        [
            "ssh",
            "-C",
            "-p",
            "16406",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "root@212.51.142.37",
            "python3 -",
        ],
        input=remote,
        capture_output=True,
        text=True,
        check=True,
    )
    snapshot = json.loads(result.stdout)
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, contents in snapshot["files"].items():
        path = args.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    write_json(
        args.destination / "check.json",
        {"checked_at_utc": snapshot["checked_at_utc"], "remote_root": ROOT, "check_error": None},
    )
    refresh(args)


def refresh(args):
    """Rebuild the offline page without inventing a new successful cloud check."""
    checked = json.loads((args.destination / "check.json").read_text())["checked_at_utc"]
    render(args.destination, args.page)
    manifest = {
        str(p.relative_to(args.destination)): sha256(p)
        for p in sorted(args.destination.rglob("*"))
        if p.is_file() and p.name != "archive.json"
    }
    write_json(
        args.destination / "archive.json",
        {
            "checked_at_utc": checked,
            "local_generated_at_utc": datetime.now(UTC).isoformat(),
            "sha256": manifest,
            "html_sha256": sha256(args.page),
            "html_path": str(args.page),
        },
    )
    for name, digest in manifest.items():
        if sha256(args.destination / name) != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")
    print(
        json.dumps(
            {
                "files": len(manifest),
                "checked_at_utc": checked,
                "page": str(args.page),
            },
            ensure_ascii=False,
        )
    )


def render(directory, page):
    protocol = json.loads((directory / "protocol.json").read_text())
    check = json.loads((directory / "check.json").read_text())
    rows = []
    run_summaries = {}
    for run in sorted((directory / "runs").glob("*")):
        status_file = run / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        valid = []
        frame_count = 0
        for seed in protocol["environment_seeds"]:
            seed_dir = run / f"seed-{seed}"
            episode, summary = completed_episode(seed_dir, seed)
            result = "未完成"
            if summary.get("input_integrity") is False:
                result = "相机冻结（不计分）"
            elif episode is not None:
                result = "成功" if episode["success"] else "失败"
            elif status.get("status") == "infrastructure_failed" and seed == status.get("seed"):
                result = "运行故障（不计策略成绩）"
            raw = summary.get("raw") or {}
            if episode is not None:
                valid.append(episode)
                frame_count += raw.get("control_frames", 0)
            control_path = seed_dir / "evidence/control.jsonl"
            controls = control_path.read_text().splitlines() if control_path.exists() else []
            completed_steps = summary.get("control_steps", len(controls) if controls else "—")
            rows.append(
                [
                    run.name,
                    seed,
                    result,
                    episode.get("episode_length", "—") if episode else completed_steps,
                    summary.get("inference_calls", "—"),
                    raw.get("control_frames", "—"),
                ]
            )
        run_summaries[run.name] = {
            "valid": len(valid),
            "successes": sum(episode["success"] for episode in valid),
            "raw_frames_per_view": frame_count,
        }
    statuses = {
        r.name: json.loads((r / "status.json").read_text())
        for r in (directory / "runs").glob("*")
        if (r / "status.json").exists()
    }
    current = (
        max(statuses, key=lambda k: statuses[k].get("updated_at_utc", "")) if statuses else None
    )
    stage = statuses[current] if current else {"stage": "deployed"}

    def esc(value):
        return html.escape(str(value))

    selected = {}
    for phase in ("rgb", "raw"):
        candidates = [name for name, status in statuses.items() if status.get("phase") == phase]
        if candidates:
            selected[phase] = max(candidates, key=lambda k: statuses[k].get("updated_at_utc", ""))

    def table_rows(values):
        return "".join(
            "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in values
        )

    table = table_rows([row for row in rows if row[0] in selected.values()])
    history = table_rows([row for row in rows if row[0] not in selected.values()])
    cards = []
    for phase, name in selected.items():
        result = run_summaries[name]
        n = result["valid"]
        score = f"{result['successes']}/{n}" if n else "未测量"
        label = "RGB 策略部署验收" if phase == "rgb" else "RGB 策略部署验收（同步采集 RAW）"
        cards.append(
            f'<div class="card"><b>{label} · {esc(name)}</b>'
            f'<p class="score">成功 {score}</p><p>有效完成 {n}/3</p>'
            f"<p>完整 RAW 帧/视角：{result['raw_frames_per_view'] or '—'}</p></div>"
        )
    verification_file = directory / "verification.json"
    verified = "联合留档核验待完成"
    details = ""
    if verification_file.exists():
        verification = json.loads(verification_file.read_text())
        verified = f"云端逐文件核验完成：{verification['verified_at_utc']}"
        details = f"<pre>{esc(json.dumps(verification, ensure_ascii=False, indent=2))}</pre>"
    conclusion = "完整联合结果与核验待完成。"
    if verification_file.exists():
        conclusion = (
            "已证明官方预训练策略可在同一成功任务中连续采集原生 CFA。"
            "RAW 种子 2 完成抬升后超时，未满足入碗判据；该失败保留。"
            "此前一次 SIGBUS 在新目录恢复原种子，底层原因尚未确定。"
            "额外渲染使两阶段像素与动作轨迹不完全相同，长期稳定性与正式比较协议仍待验证。"
        )
    text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arena DROID π0.5 验收</title>
<style>body{{font:15px system-ui;max-width:1100px;margin:30px auto;
background:#101823;color:#e5eaf1;padding:0 20px}}
h1{{font-size:25px}}
table{{width:100%;border-collapse:collapse}}
td,th{{padding:10px;text-align:left;border-bottom:1px solid #334052}}
pre{{white-space:pre-wrap;background:#182432;padding:15px}}
a{{color:#8acaff}}
.muted{{color:#a6b4c7}}
.cards{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{background:#182432;padding:16px;flex:1;min-width:230px}}
.score{{font-size:25px;margin:12px 0}}
.notice{{border-left:4px solid #e2b76c;background:#29251e;padding:14px 18px;margin:20px 0}}
</style>
<h1>Isaac Lab Arena DROID · 官方 π0.5 jointpos</h1>
<p>最后成功云端检查：{esc(check["checked_at_utc"])} · 手动检查，未建立定时任务</p>

<div class="notice"><b>当前交付未完成：完整基准与原始训练演示回放</b>
<p>① 完整九配置 RGB 基准正在独立运行，见 <a href="progress.html">当前进度</a>；
下方仅为单任务 3 种子的历史部署验收。</p>
<p>② 原始仿真专家演示的可验证回放及配对 RGB/RAW 数据集尚未生成。
现有 RAW 来自策略评测时的同步采集，不是原始训练演示回放；模型仍输入 RGB。</p>
<p>用户确认继续使用 Isaac Sim / Arena DROID 和原生 CameraCore RAW。
PolaRiS 公共仿真训练 RLDS 缺少完整物体状态；通用采集、Mimic 与回放脚本已找到，
当前 DROID 仍需专家来源及控制 / Mimic 适配。</p></div>

<p>历史部署运行：{esc(current)} · {esc(stage.get("stage"))}
 · {esc(stage.get("status", "complete"))}</p>

<p>任务：魔方入碗。预定环境种子 0、1、2；RGB 与原生 RAW 配对测试。使用官方预训练权重，无微调。</p>

<div class="cards">{"".join(cards)}</div>
<p>{esc(conclusion)}</p>
<p class="muted">这是 3 个配对种子的单任务部署验收，两阶段不能合并为 6 个独立样本。
未完成 episode 不计算成功率；运行故障不算策略失败。{esc(verified)}</p>

<table><thead><tr><th>运行版本</th>
<th>种子</th>
<th>结果</th>
<th>控制步</th>
<th>策略请求</th>
<th>双视角 RAW 帧/视角</th>
</tr></thead><tbody>{table}</tbody></table>
<p>RAW 采集协议：1280×720、GRBG、uint32 无损 CFA，保留对应线性 FP16 HDR 和原生噪声版本。
物理传感器曝光、增益及噪声尚未校准；策略使用官方 RGB。原生加噪缓冲可能超过 CFA 白电平，未做裁剪。
所有大数组与权重留在云端，本地仅数字/哈希。</p>
<details><summary>部署故障历史（不计策略分数）</summary><table>{history}</table></details>
<details><summary>最终数字、时延与留档核验</summary>{details or "待完成"}</details>

<details><summary>当前状态详情</summary>
<pre>{esc(json.dumps(stage, ensure_ascii=False, indent=2))}</pre>
</details>
<p class="muted">页面为离线快照；本地文件生成时间不代表新的云端检查。</p>
</html>"""
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("research/experiments/results/2026-10-10_pi05_arena_droid"),
    )
    parser.add_argument(
        "--page", type=Path, default=Path("outputs/arena-droid-baseline/history.html")
    )
    parser.add_argument(
        "--offline", action="store_true", help="Rebuild from existing local numbers"
    )
    args = parser.parse_args()
    refresh(args) if args.offline else archive(args)


if __name__ == "__main__":
    main()
