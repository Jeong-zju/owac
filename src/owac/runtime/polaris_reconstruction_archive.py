"""Pull small reconstruction evidence only and render an offline coverage dashboard."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

from owac.runtime.arena_droid import sha256, write_json
from owac.runtime.polaris_reconstruction import effective_status
from owac.runtime.polaris_replay import utc

REMOTE = "/workspace/owac/reference/polaris-rgb-raw-reconstruction-v1"
DESTINATION = Path("research/experiments/results/2026-10-10_polaris_full_reconstruction")
PAGE = Path("outputs/polaris-rgb-raw-reconstruction/progress.html")


def remote_script():
    return f"""import pathlib,json,hashlib,datetime,subprocess
r=pathlib.Path({REMOTE!r})
paths=[r/'inventory-v2'/n for n in ('status.json','summary.json','shards.json',
 'source_catalog.json','asset_catalogs.json','episodes.jsonl')]
paths += [r/n for n in ('view-alias-audit-v1.json','inventory-v2-recovery.json',
 'export-validation-recovery-v1.json','robot-kinematics-v1.json',
 'camera-estimator-validation-v2.json','source-object-review-v1.json',
 'grasp-assistance-authorization-v1.json','route_retirement.json')]
for camera in sorted(r.glob('camera-estimate-v*')):
 if camera.is_dir(): paths += sorted(camera.glob('*.json'))
for tracks in sorted(r.glob('object-tracks-v*')):
 if tracks.is_dir():
  paths += [tracks/n for n in ('status.json','manifest.json')]
  paths += sorted((tracks/'episodes').glob('*.json'))
for queue in sorted(r.glob('queue-v*')):
 paths += [queue/n for n in ('manifest.json','queue-status.json','scope-clarification.json',
 'source-manifest.json')]
for batch in sorted(r.glob('*-profile-batch-v*')):
 paths += [batch/n for n in ('status.json','plan.json','recovered-status-v1.json','reuse-v1.json')]
 paths += sorted((batch/'configs').glob('*.json'))
 for ep in sorted((batch/'runs').glob('*')):
  paths += [ep/'replay'/n for n in ('result.json','config.json','dataset.json','error.json',
   'postprocess.json')]
for output in sorted(r.glob('training-export-*')):
 if output.is_dir():
  paths += [output/n for n in ('coverage.json','training_loader_validation.json')]
files=[]
for p in paths:
 if p.is_file():
  if p.stat().st_size > 8000000: raise ValueError('Oversize metadata: '+str(p))
  b=p.read_bytes();files.append(dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(b).hexdigest(),text=b.decode()))
print(json.dumps(dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 files=files,supervisor=subprocess.run(['supervisorctl','status'],capture_output=True,text=True).stdout)))
"""


def counts(batch):
    rows = batch.get("completed", [])
    valid = [r for r in rows if not r["infrastructure_error"] and r["result"]]
    return {
        "completed": len(valid),
        "attempted": len(rows),
        "physically_successful": sum(bool(r["result"]["success"]) for r in valid),
        "physical_failures": sum(not r["result"]["success"] for r in valid),
        "infrastructure_failures": len(rows) - len(valid),
        "paired_control_frames": sum(r["result"]["source_frames"] for r in valid),
        "free_contact_successes": sum(
            bool(r["result"]["success"]) and not r["result"].get("grasp_assistance") for r in valid
        ),
        "constraint_assisted_successes": sum(
            bool(r["result"]["success"]) and bool(r["result"].get("grasp_assistance"))
            for r in valid
        ),
    }


def dashboard(directory, page):
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    check = read(directory / "check.json")
    retirement = read(directory / "route_retirement.json")
    route_notice = (
        "<section class='warning'><h2>路线已终止 · 后续统一使用 RAWVLA-bench</h2>"
        "<p>用户已终止 Arena DROID / PolaRiS 重建及公开发布。以下仅保留历史证据，"
        "未完成部分不再继续执行。</p>"
        "<a href='../../research/experiments/2026-10-10_rawvla_pi05_full_baseline.md'>"
        "查看 RAWVLA-bench 当前基线记录</a></section>"
        if retirement.get("status") == "retired_by_user"
        else ""
    )
    inventory = read(directory / "inventory-v2/summary.json")
    audit = read(directory / "view-alias-audit-v1.json")
    batches = []
    for path in sorted(directory.glob("*-profile-batch-v*/status.json")):
        status = effective_status(path.parent)
        plan = read(path.parent / "plan.json")
        batches.append((path.parent.name, status, plan, counts(status)))
    esc = lambda v: html.escape(str(v))  # noqa: E731
    batches.sort(
        key=lambda entry: entry[1].get("started_at_utc", entry[2].get("prepared_at_utc", ""))
    )
    latest_rows = {}
    for _, status, _, _ in batches:
        for row in status.get("completed", []):
            latest_rows[row["id"]] = row
    current = counts({"completed": list(latest_rows.values())})
    scene_rows = "".join(
        f"<tr><td>{esc(scene)}</td><td>{total}</td></tr>"
        for scene, total in inventory.get("scenes", {}).items()
    )
    trial_rows = ""
    for name, status, _, count in reversed(batches):
        trial_rows += (
            f"<h3>{esc(name)} · {esc(status.get('stage', 'pending'))}</h3>"
            f"<p>完成 {count['completed']}，物理成功 {count['physically_successful']}，"
            f"物理失败 {count['physical_failures']}，"
            f"基础设施故障 {count['infrastructure_failures']}。</p>"
            "<table><tr><th>源身份</th><th>物理结果</th><th>帧数</th><th>未满足条件</th></tr>"
        )
        for row in status.get("completed", []):
            result = row.get("result") or {}
            outcome = (
                "基础设施故障"
                if row["infrastructure_error"]
                else (
                    ("辅助重建成功" if result.get("grasp_assistance") else "自由接触成功")
                    if result.get("success")
                    else "有效失败"
                )
            )
            failed = [k for k, value in result.get("checks", {}).items() if not value]
            trial_rows += (
                f"<tr><td>{esc(row['id'])}</td><td>{outcome}</td>"
                f"<td>{result.get('source_frames', '—')}</td><td>{esc(', '.join(failed))}</td></tr>"
            )
        trial_rows += "</table>"
    loader = []
    for path in sorted(directory.glob("training-export-*/coverage.json")):
        coverage = read(path)
        loader.append(
            f"<p>{esc(path.parent.name)}：{coverage['episodes']} 条 / "
            f"{coverage['source_frames']} 帧；实际训练加载验证 "
            f"{esc(coverage['training_loader_validated'])}；"
            f"RAW 整数精确检查 {coverage['raw_exact_value_checks']} 次。</p>"
        )
    errors = (
        f"<p class='warning'>检查失败：{esc(check['check_error'])}</p>"
        if check.get("check_error")
        else ""
    )
    camera_detail = ""
    for path in sorted(directory.glob("camera-estimate-v*/status.json")):
        camera = read(path)
        camera_detail += (
            f"<p>{esc(path.parent.name)}：已检查 {camera.get('checked_scenes', 0)} / "
            f"{camera.get('source_scenes', 14)} 个源场景；"
            f"候选外部机位 {camera.get('candidate_estimates', 0)} 个。"
            "基于源 RGB 与机器人位姿拟合，须再检查重建画面；原始标定未恢复。</p>"
        )
    object_detail = ""
    for path in sorted(directory.glob("object-tracks-v*/status.json")):
        tracks = read(path)
        object_detail += (
            f"<p>{esc(path.parent.name)}：已分析 {tracks.get('checked_episodes', 0)} / "
            f"{tracks.get('source_episodes', 316)} 条源轨迹；"
            f"推断出 {tracks.get('candidate_object_count', 0)} 个物体轨迹候选，"
            f"{tracks.get('unresolved_grasps', 0)} 个抓取段需复核。"
            "候选数量不是已恢复物体状态或已建成场景数量。</p>"
        )
    queue_detail = ""
    for path in sorted(directory.glob("queue-v*/manifest.json")):
        queue = read(path.parent / "queue-status.json")
        queued = read(path)
        clarification = read(path.parent / "scope-clarification.json")
        queue_detail += (
            f"<p>串行云端任务 {esc(path.parent.name)}：{esc(queue.get('status', '尚未启动'))}；"
            f"已完成批次 {len(queue.get('completed_batches', []))} / "
            f"{len(queued.get('batches', []))}。"
            f"当前：<code>{esc(queue.get('active_batch', queue.get('stage', '—')))}</code></p>"
            f"<p>{esc(queued.get('scope', ''))}</p>"
        )
        if clarification:
            queue_detail += (
                f"<p>旋转任务源轨迹 {clarification['original_rotation_sources']} 条，"
                f"可开始原生采集 {clarification['ready_for_native_trial']} 条，"
                "另 1 条需复核抓取/旋转锚点，不计为已完成。</p>"
            )
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PolaRiS 原生 RGB / RAW 全集重建</title><style>
body{{font:16px system-ui;background:#0b1220;color:#e2e8f0;
max-width:1200px;margin:auto;padding:28px}}
h1{{font-size:28px}}a{{color:#7dd3fc}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}
.card,section{{background:#152036;border:1px solid #334155;
border-radius:12px;padding:20px;margin:12px 0}}
.card{{flex:1;min-width:180px}}strong{{font-size:30px;display:block}}.warning{{color:#fbbf24}}
table{{width:100%;border-collapse:collapse}}
td,th{{padding:9px;text-align:left;border-bottom:1px solid #334155}}
code{{overflow-wrap:anywhere}}small{{color:#94a3b8}}</style>
<h1>PolaRiS · 全集 RGB / 原生 RAW 重建</h1>
{route_notice}
<p class="warning">尚未完成全集，尚未公开上传。派生重建数据不称为官方原场景精确补采。</p>
<p>最近成功云端检查：<time>{esc(check.get("checked_at_utc", "尚未检查"))}</time>。
定时任务关闭。</p>{errors}
{queue_detail}
<div class="cards"><div class="card">源轨迹审计<strong>{inventory.get("episodes", 0)} / 316</strong>
{inventory.get("frames", 0):,} 个原控制帧</div><div class="card">完整配对回放（各身份最新版本）
<strong>{current.get("completed", 0)} / 316</strong>全部三路字段，两个实际相机</div>
<div class="card">通过重建任务检查<strong>{current.get("physically_successful", 0)} / 316</strong>
自由接触 {current.get("free_contact_successes", 0)}，
约束辅助 {current.get("constraint_assisted_successes", 0)}；
还须逐条场景/RGB 对应审查</div></div>
<section><h2>完整范围与对应关系</h2><p>保留每条源身份、原动作、源 RGB 参考；逐任务重建物体和场景，
用 Isaac Sim 6.1 CameraCore 在 ISP 前采 GRBG uint32 CFA，与同帧新 RGB 并列。
没有恢复原相机标定、背景和精确像素对齐；这些缺失不能由成功计数代替。</p>
<p>审计 {audit.get("frames", 0):,} 个帧的两个外部 JPEG 字段完全相同，
因此 exterior 2 明确别名到 exterior 1；腕部为独立相机。保持原三字段结构。</p>
<p>完整交付门槛：316 条身份、52,914 帧、三路配对字段、全部任务物理检查、
逐条源 RGB 对应审查、实际 LeRobot / π0.5 加载与原 RAW 整数精确校验。</p>
<p>目标公开仓库：<code>Jeong-zju/isaac-arena-droid-rgb-raw-reconstructed</code>，尚未创建。</p></section>
<section><h2>重建批次与失败记录</h2>{trial_rows or "<p>尚无云端回放批次。</p>"}</section>
<section><h2>实际训练格式验证</h2>{"".join(loader) or "<p>尚未通过实际训练加载器。</p>"}</section>
<section><h2>源 RGB 对应诊断</h2>{camera_detail or "<p>相机拟合尚未完成。</p>"}
{object_detail}
<p>用户已于 2026-10-11 接受原夹爪动作触发抓取约束。辅助重建采用当前位置建立固定关节，
原松手动作解除；动作不改、物体不瞬移。须记录接近距离、启停事件和约束误差，
与自由接触结果分报；该方法不用于冻结的 π0.5 基准。</p>
<p>源 marker 任务存在多物体清理；单物体原型缺少其余物体，不能作为完整专家场景。</p>
<p>关节命令回放与源状态驱动回放按独立版本保存；后者保留原动作标签，
但机械臂由源关节状态驱动，不能冒称执行原机械臂动作命令。</p></section>
<section><h2>源场景覆盖</h2><table><tr><th>源场景</th><th>轨迹</th></tr>{scene_rows}</table></section>
<p><a href="../arena-droid-baseline/progress.html">π0.5 完整 RGB 基准</a> ·
<a href="../../research/experiments/2026-10-10_polaris_full_reconstruction.md">实验记录</a> ·
<a href="../../research/experiments/results/2026-10-10_polaris_full_reconstruction/check.json">
检查数字</a></p>
<small>此页可离线打开。图像、RAW、TFRecord、检查点均留在云端；页面生成不是新的远端检查。</small></html>""")


def archive(destination, page):
    destination.mkdir(parents=True, exist_ok=True)
    check_path = destination / "check.json"
    check = json.loads(check_path.read_text()) if check_path.exists() else {}
    try:
        response = subprocess.run(
            [
                "ssh",
                "-C",
                "-p",
                "16406",
                "-o",
                "ConnectTimeout=20",
                "root@212.51.142.37",
                "python3 -",
            ],
            input=remote_script(),
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        # SSH may emit the host's welcome banner before the JSON response.
        snapshot = json.loads(response.stdout[response.stdout.index('{"checked_at_utc"') :])
        for entry in snapshot["files"]:
            content = entry["text"].encode()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("Transferred numerical evidence failed checksum")
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe remote metadata path")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        check = {
            "checked_at_utc": snapshot["checked_at_utc"],
            "remote": REMOTE,
            "remote_files": len(snapshot["files"]),
            "supervisor": snapshot["supervisor"],
            "full_dataset_complete": False,
            "public_upload_complete": False,
            "route_status": (
                json.loads((destination / "route_retirement.json").read_text())["status"]
                if (destination / "route_retirement.json").exists()
                else "active"
            ),
        }
    except Exception as error:
        check["check_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(check_path, check)
        dashboard(destination, page)
        files = {
            str(p.relative_to(destination)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(destination.rglob("*"))
            if p.is_file() and p.name != "archive.json"
        }
        write_json(
            destination / "archive.json",
            {
                "remote_checked_at_utc": check.get("checked_at_utc"),
                "local_generated_at_utc": utc(),
                "files": files,
                "page": {"path": str(page), "sha256": sha256(page)},
            },
        )
        for name, metadata in files.items():
            if sha256(destination / name) != metadata["sha256"]:
                raise ValueError("Local numerical archive changed after generation")
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--page", type=Path, default=PAGE)
    args = parser.parse_args()
    result = archive(args.destination, args.page)
    print(json.dumps({k: v for k, v in result.items() if k != "supervisor"}))


if __name__ == "__main__":
    main()
