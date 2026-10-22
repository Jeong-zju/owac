"""Local Chinese monitor for all-bench prepare/training JSONL."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

TOTAL_SOURCES = 2163
TOTAL_ROWS = 74129
TOTAL_STEPS = 10596


def _latest(path: Path) -> dict:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else {}


def _prepare(root: Path) -> tuple[int, int]:
    path = root / "prepare-progress.jsonl"
    unique: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                unique[str(row.get("source_id", ""))] = row
            except (json.JSONDecodeError, TypeError):
                continue
    return len(unique), sum(int(row.get("rows", 0)) for row in unique.values())


def _page(root: Path) -> str:
    train = _latest(root / "updates.jsonl")
    val = _latest(root / "validation.jsonl")
    status = (
        json.loads((root / "run-status.json").read_text())
        if (root / "run-status.json").is_file()
        else {}
    )
    completed, rows = _prepare(root)
    controller = str(status.get("status", ""))
    native_exit = status.get("native_exit")
    failed = controller == "failed" or (native_exit not in (None, 0, "0"))
    if failed:
        reason = str(status.get("error") or status.get("reason") or "训练进程失败")
        phase = f"已停止：{reason}"
    elif controller == "preparing" or completed < TOTAL_SOURCES:
        phase = "正在生成教师目标"
    elif not train:
        phase = "学生尚未训练"
    elif controller == "completed":
        phase = "学生训练已完成"
    elif controller == "paused":
        phase = "学生训练已暂停"
    else:
        phase = "学生训练中"
    loss = train.get("loss_components", {}) if train else {}
    startup = (val.get("startup") or {}) if val else {}
    eligible = (val.get("eligible_ge_25") or {}) if val else {}
    update = train.get("new_global_updates", 0) if train else 0
    epoch = train.get("epoch", 0) if train else 0
    sidecar_path = root / "swanlab-sidecar-status.json"
    sidecar_url = ""
    if sidecar_path.is_file():
        try:
            sidecar_url = str(json.loads(sidecar_path.read_text()).get("url") or "")
        except (OSError, json.JSONDecodeError):
            sidecar_url = ""

    def num(value):
        return f"{value:.6f}" if isinstance(value, (int, float)) else "-"

    train_mse = num(loss.get("mse"))
    train_cos = num(loss.get("cosine"))
    train_total = num(loss.get("total"))
    start_mse = num(startup.get("feature_mse"))
    eligible_mse = num(eligible.get("feature_mse"))
    val_mse = num(val.get("feature_mse"))
    val_cos = num(val.get("feature_cosine"))
    val_total = num(val.get("feature_total"))
    return f"""<html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='5'><title>OWAC 全量蒸馏</title></head>
<body style='font-family:sans-serif;max-width:900px;margin:2em auto'>
<h2>OWAC 全量 RawVLA-Bench 蒸馏</h2>
<h3>{phase}</h3>
<p>训练与验证轨迹: {completed}/{TOTAL_SOURCES}，观测时刻: {rows}/{TOTAL_ROWS}</p>
<p>第 {epoch}/3 遍训练集，更新 {update}/{TOTAL_STEPS}</p>
<p>训练MSE {train_mse} / 方向误差 {train_cos} / 总损失 {train_total}</p>
<p>验证启动阶段MSE {start_mse}；验证后续阶段MSE {eligible_mse}</p>
<p>验证整体MSE {val_mse} / 方向误差 {val_cos} / 总损失 {val_total}</p>
<p><a href='{sidecar_url}'>SwanLab 新运行</a></p>
<p>磁盘上限 380 GiB；无 GPU 时间/显存/利用率停止门槛。</p>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8880)
    parser.add_argument(
        "--run-root", default="/root/owac/repo/outputs/raw-distillation-all-bench-20261022"
    )
    args = parser.parse_args()
    root = Path(args.run_root)
    if not args.serve:
        print(_page(root))
        return 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = _page(root).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
