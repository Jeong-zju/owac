"""Run the frozen nine-variation Arena benchmark and archive small numerical results."""

from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.arena_droid_verify import verify_phase, wilson

REMOTE = "/workspace/owac/reference/pi05-arena-droid-benchmark-v1"
SMALL_FILES = (
    "protocol.json",
    "status.json",
    "summary.json",
    "episodes.jsonl",
    "source_hashes.json",
    "replay_audit.json",
    "deployment_validation.json",
)


def identities(protocol):
    variations = protocol["variations"]
    seeds = protocol["environment_seeds"]
    names = [v["name"] for v in variations]
    if not names or len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate or empty benchmark identities")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Invalid environment seed")
    pairs = [(name, seed) for seed in seeds for name in names]
    if len(pairs) != protocol["expected_episodes"]:
        raise ValueError("Expected episode count differs from the frozen identities")
    return pairs


def summarize(protocol, rows):
    expected = set(identities(protocol))
    seen = [(row["variation"], row["seed"]) for row in rows]
    if len(set(seen)) != len(seen) or not set(seen) <= expected:
        raise ValueError("Duplicated or undeclared scored episode")
    if any(type(row["success"]) is not bool for row in rows):
        raise ValueError("A scored episode must have an explicit boolean outcome")
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    return {
        "complete": count == len(expected),
        "episodes": count,
        "expected_episodes": len(expected),
        "successes": successes,
        "success_rate": successes / count if count else None,
        "wilson_95": wilson(successes, count) if count else None,
        "per_variation": [
            {
                "name": variation["name"],
                "episodes": len(selected),
                "expected": len(protocol["environment_seeds"]),
                "successes": sum(row["success"] for row in selected),
                "wilson_95": wilson(sum(row["success"] for row in selected), len(selected))
                if selected
                else None,
            }
            for variation in protocol["variations"]
            for selected in [[r for r in rows if r["variation"] == variation["name"]]]
        ],
    }


def validate_snapshot(protocol, status, reported, rows):
    """Verify cloud totals against the complete set of independently scored identities."""
    computed = summarize(protocol, rows)
    for key, value in computed.items():
        if reported.get(key) != value:
            raise ValueError(f"Cloud summary disagrees with episode records: {key}")
    if status["completed_episodes"] != computed["episodes"]:
        raise ValueError("Cloud status disagrees with episode records")
    if status["status"] == "complete" and not computed["complete"]:
        raise ValueError("Cloud completion lacks the full frozen episode set")
    if rows:
        if reported["inference_calls"] != sum(row["inference_calls"] for row in rows):
            raise ValueError("Cloud inference count disagrees with episode records")
        timings = [reported[f"inference_roundtrip_p{p}_ms"] for p in (50, 95)]
        if not all(math.isfinite(x) and x > 0 for x in timings) or timings[1] < timings[0]:
            raise ValueError("Invalid cloud latency summary")
    return reported


def run(args):
    protocol = json.loads(args.protocol.read_text())
    pairs = identities(protocol)
    if protocol["raw_in_evaluation"] or protocol["fine_tuning"]:
        raise ValueError("This protocol evaluates the unchanged official RGB policy")
    weights = args.runtime_root / "checkpoints/pi05_droid_jointpos_pytorch"
    for path, key in (
        (weights / "model.safetensors", "model_sha256"),
        (weights / "assets/droid/norm_stats.json", "normalization_sha256"),
        (args.runtime_root / "Arena" / protocol["upstream_config"], "upstream_config_sha256"),
    ):
        if sha256(path) != protocol[key]:
            raise ValueError(f"Frozen identity mismatch: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "replay_audit.json", json.loads(args.replay_audit.read_text()))
    source = Path(__file__).resolve().parents[1]
    write_json(
        args.output / "source_hashes.json",
        {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))},
    )
    write_json(
        args.output / "deployment_validation.json",
        {
            "validated_at_utc": datetime.now(UTC).isoformat(),
            "runtime_root": str(args.runtime_root),
            "model_sha256": protocol["model_sha256"],
            "normalization_sha256": protocol["normalization_sha256"],
            "upstream_config_sha256": protocol["upstream_config_sha256"],
            "policy_changes": "none; only numerical latency logging added",
        },
    )
    rows = []
    latencies = []
    status = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "infrastructure_failures": [],
        "stage": "full_rgb_benchmark",
    }
    started = time.monotonic()

    def save():
        status.update(
            updated_at_utc=datetime.now(UTC).isoformat(),
            completed_episodes=len(rows),
            wall_seconds=time.monotonic() - started,
        )
        summary = summarize(protocol, rows)
        if latencies:
            import numpy as np

            summary.update(
                inference_roundtrip_p50_ms=float(np.percentile(latencies, 50)) * 1000,
                inference_roundtrip_p95_ms=float(np.percentile(latencies, 95)) * 1000,
                inference_calls=len(latencies),
            )
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "status.json", status)

    save()
    try:
        for name, seed in pairs:
            variation = next(v for v in protocol["variations"] if v["name"] == name)
            episode_root = args.output / "runs" / name / f"seed-{seed}"
            episode_root.mkdir(parents=True, exist_ok=False)
            episode_protocol = {**protocol, **variation, "environment_seeds": [seed]}
            write_json(episode_root / "protocol.json", episode_protocol)
            for attempt in (1, 2):
                output = episode_root / f"attempt-{attempt:02d}"
                status.update(variation=name, seed=seed, attempt=attempt)
                save()
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "owac.runtime.arena_droid",
                    "run",
                    "--root",
                    str(args.runtime_root),
                    "--output",
                    str(output),
                    "--protocol",
                    str(episode_root / "protocol.json"),
                    "--phase",
                    "rgb",
                    "--port",
                    str(args.port),
                ]
                write_json(
                    episode_root / f"attempt-{attempt:02d}-command.json",
                    {
                        "argv": command,
                        "retry_reason": status["infrastructure_failures"][-1]
                        if attempt > 1
                        else None,
                    },
                )
                try:
                    with (episode_root / f"attempt-{attempt:02d}.log").open("w") as log:
                        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                    verified = verify_phase(output, [seed])
                    timing = [
                        json.loads(line)["seconds"]
                        for line in (output / f"seed-{seed}/evidence/inference.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    if len(timing) != verified["trials"][0]["inference_calls"]:
                        raise ValueError("Inference timings are incomplete")
                    if any(not math.isfinite(x) or x <= 0 for x in timing):
                        raise ValueError("Invalid measured latency")
                    row = {
                        "variation": name,
                        "attempt": attempt,
                        "evidence": str(output.relative_to(args.output)),
                        "official_outcome_files": {
                            str(p.relative_to(args.output)): sha256(p)
                            for p in (output / f"seed-{seed}/official").rglob(
                                "episode_results*.jsonl"
                            )
                        },
                        "control_sha256": sha256(output / f"seed-{seed}/evidence/control.jsonl"),
                        "inference_sha256": sha256(
                            output / f"seed-{seed}/evidence/inference.jsonl"
                        ),
                        **verified["trials"][0],
                    }
                    rows.append(row)
                    latencies.extend(timing)
                    with (args.output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    save()
                    break
                except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as error:
                    status["infrastructure_failures"].append(
                        {
                            "variation": name,
                            "seed": seed,
                            "attempt": attempt,
                            "error": repr(error),
                            "at_utc": datetime.now(UTC).isoformat(),
                        }
                    )
                    save()
                    if attempt == 2:
                        raise
            print("ARENA_BENCHMARK_EPISODE", name, seed, row["success"], len(rows), flush=True)
        status.update(status="complete", stage="full_rgb_benchmark_complete")
        save()
    except BaseException as error:
        status.update(status="infrastructure_failed", error=repr(error))
        save()
        raise


def render(directory, page):
    protocol = json.loads((directory / "protocol.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    rows = (
        [
            json.loads(x)
            for x in (directory / "episodes.jsonl").read_text().splitlines()
            if x.strip()
        ]
        if (directory / "episodes.jsonl").exists()
        else []
    )
    summary = validate_snapshot(
        protocol, status, json.loads((directory / "summary.json").read_text()), rows
    )
    check = json.loads((directory / "check.json").read_text())
    audit = json.loads((directory / "replay_audit.json").read_text())
    pilot_path = directory / "replay_pilot.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    esc = lambda x: html.escape(str(x))  # noqa: E731
    n = summary["episodes"]
    score = f"{summary['successes']}/{n}" if n else "未测量"
    interval = lambda v: f"{v[0]:.2%}–{v[1]:.2%}" if v else "未测量"  # noqa: E731
    latency = (
        f"{summary['inference_roundtrip_p50_ms']:.2f} / "
        f"{summary['inference_roundtrip_p95_ms']:.2f} ms"
        if n
        else "未测量"
    )
    memory = max((r["policy_peak_allocated_bytes"] for r in rows), default=0) / 2**30
    check_error = (
        f'<p class="notice">最近检查失败：{esc(check["check_error"])}；保留上次成功快照。</p>'
        if check.get("check_error")
        else ""
    )
    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(v)}</td>"
            for v in (
                r["name"].removeprefix("droid_pnp_srl_openpi_"),
                f"{r['episodes']}/{r['expected']}",
                f"{r['successes']}/{r['episodes']}" if r["episodes"] else "未测量",
                interval(r["wilson_95"]),
            )
        )
        + "</tr>"
        for r in summary["per_variation"]
    )
    replay_score = "尚未生成配对数据集"
    replay_status = esc(audit["status"])
    replay_detail = ""
    if pilot:
        successes = pilot["successful_unique_source_episodes"]
        replay_score = (
            f"重建回放成功专家：{successes} 条" if successes else "专家回放试验进行中 / 未成功"
        )
        replay_status = "原始专家动作 · 重建物体场景 · Isaac 原生 CFA"
        replay_detail = (
            '<div class="notice"><b>专家回放试验</b>'
            f"<p>最后成功检查：{esc(pilot['checked_at_utc'])}。"
            "原训练场景及相机映射尚未恢复；新 RGB/RAW 是重建场景中的配对观测，"
            "原动作和源 episode 身份保留。该试验不计入 π0.5 基准分数。</p>"
            f"<details><summary>回放指标与采集核验</summary><pre>"
            f"{esc(json.dumps(pilot['results'], ensure_ascii=False, indent=2))}"
            "</pre></details></div>"
        )
        successful_replays = [r for r in pilot["results"] if r["success"]]
        if successful_replays:
            trial = successful_replays[-1]
            replay_status += f" · {trial['source_frames']} 帧 × 2 视角 · 原动作未修改"
            replay_detail = (
                '<div class="card"><b>最近成功回放的物理结果</b>'
                f"<p>方块抬升 {trial['maximum_lift_m'] * 100:.2f} cm；"
                f"搬运 {trial['transport_distance_m'] * 100:.2f} cm；"
                f"最终箱中心 XY 误差 {trial['final_bin_xy_error_m'] * 100:.2f} cm。"
                "已松手并稳定入箱。</p>"
                f"<p>{trial['verified_file_count']} 个文件哈希全部通过；原生 HDR→CFA "
                f"最大量化误差 {trial['native_capture']['maximum_quantization_error_dn']} DN。</p>"
                "<p>这是一个成功源 episode 的功能验收，不作为独立成功率或完整基准。</p>"
                f"<p>云端数据目录：<code>{esc(trial['remote_directory'])}</code></p></div>"
                + replay_detail
            )
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arena DROID 两阶段进度</title><style>
body{{font:15px system-ui;background:#101823;color:#e5eaf1;max-width:1100px;
margin:30px auto;padding:0 20px}}a{{color:#8acaff}}h1{{font-size:25px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#182432;padding:18px;flex:1}}
.score{{font-size:28px}}table{{width:100%;border-collapse:collapse}}
td,th{{text-align:left;padding:10px;border-bottom:1px solid #334052}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.notice{{padding:16px;background:#29251e}}
</style><h1>Isaac Sim · Arena DROID · 官方 π0.5</h1>
<p>最后成功云端检查：{esc(check["checked_at_utc"])} · 手动快照，定时任务关闭</p>
{check_error}
<div class="cards"><div class="card"><b>阶段 1：完整 RGB 基准</b>
<p class="score">完成 {n}/{protocol["expected_episodes"]} · 成功 {score}</p>
<p>{esc(status["status"])} · {esc(status.get("variation", ""))}
 · seed {esc(status.get("seed", "—"))}</p>
<p>累计有效样本 Wilson 95%：{interval(summary["wilson_95"])}</p></div>
<div class="card"><b>阶段 2：原始专家训练数据回放</b>
<p class="score">{replay_score}</p>
<p>{replay_status}</p></div></div>
<p>固定官方九组对象/容器/HDR 配置，每组种子 0–99，共 900 个独立完整 episode。
复用官方权重，额外训练 0 步；每次重新启动策略及仿真，保留全部正常失败。
这是 OWAC 固定协议，不是整个 Arena 平台的官方总分。</p>
<p>当前统计包含已完成样本，全部 900 条完成前不作为最终基准结论。
推理请求 p50 / p95：{latency}（含冷启动）；已完成 episode 策略显存峰值：
{f"{memory:.3f} GiB" if n else "未测量"}（不含模拟器）。
流程累计 wall：{status["wall_seconds"]:.1f} s。</p>
<p>基础设施异常尝试：{len(status["infrastructure_failures"])}，不计策略失败；
只在新 attempt 目录重试同一配置 / 种子，原因保留于下方状态记录。</p>
<table><thead><tr><th>配置</th><th>有效完成</th><th>成功数</th><th>Wilson 95%</th></tr></thead>
<tbody>{table}</tbody></table>
{replay_detail}
<div class="notice"><b>原始训练回放的缺口</b><p>{esc(audit["reason"])}</p>
<p>仅采用 Isaac 原生 CameraCore 的 ISP 前 CFA；不采用 MuJoCo，
不从 RGB 逆处理或重新马赛克，不把评测策略轨迹当成原始专家训练演示。</p></div>
<p>历史单任务验收 RGB 3/3、同步 RAW 2/3 保留于
<a href="history.html">历史记录</a>；它们不计入本次 900 条基准。</p>
<details><summary>逐 episode 结果</summary>
<pre>{esc(json.dumps(rows, ensure_ascii=False, indent=2))}
</pre></details><details><summary>协议与当前状态</summary>
<pre>{esc(json.dumps({"protocol": protocol, "status": status}, ensure_ascii=False, indent=2))}
</pre></details><p>大数组与权重留在云端。页面生成时间不代表新的云端检查。</p></html>""")


def archive(args):
    if not args.offline:
        script = f"""import json,datetime,pathlib,time
r=pathlib.Path({REMOTE!r})
for attempt in range(5):
 before=(r/'status.json').read_text()
 files={{}}
 for name in {SMALL_FILES!r}:
  p=r/name
  if p.exists():
   if p.stat().st_size > 8_000_000: raise ValueError("Unexpected archive size")
   files[name]=p.read_text()
 count=len([x for x in files.get('episodes.jsonl','').splitlines() if x.strip()])
 if (before==(r/'status.json').read_text()
     and json.loads(files['summary.json'])['episodes']==count
     and json.loads(before)['completed_episodes']==count): break
 time.sleep(0.05)
else: raise ValueError("Cloud snapshot changed during archive read")
print(json.dumps({{"checked_at_utc":datetime.datetime.now(datetime.UTC).isoformat(),
                  "files":files}}))
"""
        try:
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
                input=script,
                capture_output=True,
                check=True,
                text=True,
            )
            snapshot = json.loads(result.stdout)
            files = snapshot["files"]
            validate_snapshot(
                json.loads(files["protocol.json"]),
                json.loads(files["status.json"]),
                json.loads(files["summary.json"]),
                [json.loads(x) for x in files.get("episodes.jsonl", "").splitlines() if x.strip()],
            )
        except (subprocess.CalledProcessError, ValueError, KeyError) as error:
            previous = args.destination / "check.json"
            if not previous.exists():
                raise
            check = json.loads(previous.read_text())
            check.update(check_error=repr(error), attempted_at_utc=datetime.now(UTC).isoformat())
            write_json(previous, check)
        else:
            args.destination.mkdir(parents=True, exist_ok=True)
            for name, contents in files.items():
                (args.destination / name).write_text(contents)
            write_json(
                args.destination / "check.json",
                {
                    "checked_at_utc": snapshot["checked_at_utc"],
                    "remote_root": REMOTE,
                    "check_error": None,
                },
            )
    render(args.destination, args.page)
    check = json.loads((args.destination / "check.json").read_text())
    manifest = {
        p.name: sha256(p)
        for p in args.destination.iterdir()
        if p.is_file() and p.name != "archive.json"
    }
    write_json(
        args.destination / "archive.json",
        {
            **check,
            "local_generated_at_utc": datetime.now(UTC).isoformat(),
            "sha256": manifest,
            "html_sha256": sha256(args.page),
            "html_path": str(args.page),
        },
    )
    for name, digest in manifest.items():
        if sha256(args.destination / name) != digest:
            raise ValueError(f"Archive checksum mismatch: {name}")
    print(json.dumps({"page": str(args.page), "files": len(manifest), **check}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("run")
    evaluation.add_argument("--runtime-root", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--protocol", type=Path, required=True)
    evaluation.add_argument("--replay-audit", type=Path, required=True)
    evaluation.add_argument("--port", type=int, default=18030)
    backup = commands.add_parser("archive")
    backup.add_argument(
        "--destination",
        type=Path,
        default=Path("research/experiments/results/2026-10-10_pi05_arena_droid_benchmark"),
    )
    backup.add_argument(
        "--page", type=Path, default=Path("outputs/arena-droid-baseline/progress.html")
    )
    backup.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    {"run": run, "archive": archive}[args.command](args)


if __name__ == "__main__":
    main()
