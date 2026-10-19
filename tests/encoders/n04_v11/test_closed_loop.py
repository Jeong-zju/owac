"""N04 v1.1 闭环执行器的假客户端单测。

全部用**假仿真客户端**（真实进程、真实 argv、真实落盘的 ledger／视频）驱动执行器，
不用假结果：fake client 读执行器生成的子清单、按 ``(task_id, episode_idx)`` 续跑、
按光照域写成功／失败、按 attempt 目录决定崩溃时机，执行器只做它该做的事。

覆盖：identity 去重与续跑、三种状态与分母口径、progress.json 字段与原子写、
基础设施错误重试一次但保留记录、``--force`` 行为、RoboTwin 单条与视频归档。
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from pathlib import Path

import pytest

from owac.evaluation.n04_v11_closed_loop import (
    MAX_ATTEMPTS,
    PLATFORMS,
    PROGRESS_FIELDS,
    LoopSpec,
    build_libero_client,
    build_robotwin_eval,
    load_eval_list,
    run_campaign,
)

# 只保留插桩锚点的最小作者源码：锚点漂移必须直接报错，而不是静默生成错客户端。
FAKE_LIBERO_SOURCE = """from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path

import numpy as np


def run(args: EvalArgs) -> dict[str, Any]:
    for task_id in range(1):
        payload = {"examples": [model_input]}
        for ep_idx in range(1):
            action_plan: collections.deque[np.ndarray] = collections.deque()
            client.reset()
            while t < max_steps:
                example, _ = preprocess_env_obs(obs, task_description, args.resize_size)
                obs, _, done, _ = env.step(action_plan.popleft().tolist())
        actions = postprocess_openpi_actions(
            normalized_actions=normalized_actions,
            raw_state=raw_state,
            norm_stats=self.norm_stats,
            model_name=self.model_name,
            model_source=self.model_source,
        )
            _append_episode(
                args.resume_path,
                {
                    "task_suite": args.task_suite,
                    "rawvla_bench_entry": rawvla_entry,
                },
            )
            task_sr = task_successes / max(task_episodes, 1)
"""

FAKE_ROBOTWIN_SOURCE = """import sys
import os
import subprocess

sys.path.append("./")


def main(usr_args):
    topk = 1
    args['task_name'] = task_name
    model = get_model(usr_args)
    while succ_seed < test_num:
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            # Persist exactly the pre-action observation consumed by the
            # policy, yielding one observation for every predicted action.
            break
        if eval_seeds:
            now_seed = eval_seeds[now_id]
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            except UnStableError as e:
                continue
            except Exception as e:
                continue
        if use_dataset_instructions:
            instruction_path = None
"""

FAKE_LIBERO_CLIENT = '''\
"""假 LIBERO 客户端：读执行器生成的子清单，写官方形状的 ledger 行与视频。"""

import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task-suite")
parser.add_argument("--task-start", type=int, default=0)
parser.add_argument("--max-tasks", type=int, default=1)
parser.add_argument("--rawvla-bench-manifest", type=Path)
parser.add_argument("--resume-path", type=Path)
parser.add_argument("--result-json", type=Path)
parser.add_argument("--image-mode")
parser.add_argument("--rawvla-bench-representation")
parser.add_argument("--host")
parser.add_argument("--port")
parser.add_argument("--model")
parser.add_argument("--model-source")
parser.add_argument("--no-log-action-stats", action="store_true")
args, _unknown = parser.parse_known_args()

mode = os.environ.get("FAKE_LIBERO_MODE", "ok")
attempt = int(args.resume_path.parent.name.removeprefix("attempt") or 1)
plan = json.loads(args.rawvla_bench_manifest.read_text(encoding="utf-8"))["entries"]
done = set()
if args.resume_path.exists():
    for line in args.resume_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done.add((int(record["task_id"]), int(record["episode_idx"])))

if mode == "crash" or (mode == "crash_attempt1" and attempt == 1):
    print("fake client crash on attempt", attempt)
    sys.exit(3)

args.resume_path.parent.mkdir(parents=True, exist_ok=True)
fail_domains = set(filter(None, os.environ.get("FAKE_FAIL_DOMAINS", "").split(",")))
video_dir = os.environ.get("RAWVLA_VIDEO_DIR")
rows = []
with args.resume_path.open("a", encoding="utf-8") as stream:
    for index, entry in enumerate(plan):
        key = (int(entry["task_id"]), index)
        if key in done:
            continue
        if mode == "partial_attempt1" and attempt == 1 and index >= 1:
            stream.flush()
            sys.exit(3)
        success = entry["lighting_domain"] not in fail_domains
        record = {
            "task_suite": args.task_suite,
            "task_id": int(entry["task_id"]),
            "episode_idx": index,
            "success": success,
            "steps": 5 if success else 6,
            "seconds": 1.5,
            "max_steps": 6,
            "flow_seed": 4242,
            "rawvla_bench_entry": entry,
        }
        stream.write(json.dumps(record) + "\\n")
        stream.flush()
        os.fsync(stream.fileno())
        rows.append(record)
        if video_dir:
            identity = (f"libero:{args.task_suite}:{entry['task_id']}:"
                        f"{entry['init_state_index']}:{entry['lighting_domain']}")
            target = Path(video_dir) / f"{identity}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake-mp4")
if args.result_json:
    args.result_json.write_text(json.dumps({"episodes": len(rows)}) + "\\n")
print("fake libero client wrote", len(rows), "rows")
'''

FAKE_ROBOTWIN_CLIENT = '''\
"""假 RoboTwin 客户端：写作者形状的 episode_results.jsonl 与 episode0.mp4。"""

import argparse
import json
import os
from pathlib import Path

# 与作者 eval_policy.py 相同的严格 CLI：未知参数在 --overrides 之前会直接报错退出，
# 这样命令拼接顺序错了测试就会失败（防止 5 个身份被误判）。
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--policy_ckpt_path", default=None)
parser.add_argument("--overrides", nargs=argparse.REMAINDER)
args = parser.parse_args()
overrides = {}
if args.overrides:
    for i in range(0, len(args.overrides), 2):
        overrides[args.overrides[i].lstrip("--")] = args.overrides[i + 1]
for required in ("task_name", "task_config", "ckpt_setting", "policy_name", "seed"):
    if required not in overrides:
        raise SystemExit(f"缺少 overrides 键 {required}: {args.overrides}")
args.task_name = overrides["task_name"]
args.task_config = overrides["task_config"]

success = os.environ.get("FAKE_ROBOTWIN_SUCCESS", "1") == "1"
# 作者的 eval_policy.py 把视频写在 cwd 的 eval_result/... 下，把结果行写在审计目录。
video = Path.cwd() / "eval_result" / args.task_name / "episode0.mp4"
video.parent.mkdir(parents=True, exist_ok=True)
video.write_bytes(b"fake-robotwin-mp4")
record = {
    "task": args.task_name,
    "eval_seed": int(os.environ["ROBOTWIN_EVAL_SEEDS"]),
    "success": success,
    "num_steps": 9,
    "seconds": 2.5,
    "timeout": False,
}
audit_dir = Path(os.environ["ROBOTWIN_POLICY_AUDIT_DIR"])
audit_dir.mkdir(parents=True, exist_ok=True)
(audit_dir / "episode_results.jsonl").write_text(json.dumps(record) + "\\n", encoding="utf-8")
print("fake robotwin client wrote result", success)
'''


class DummyPolicyService:
    """只监听端口的假策略服务：执行器 attach 模式需要它存在（不实现策略协议）。"""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(8)
        self.port = int(self._socket.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._socket.accept()
            except OSError:
                return
            connection.close()

    def close(self) -> None:
        self._stop.set()
        self._socket.close()


def _fake_clients(tmp_path: Path) -> dict[str, Path]:
    libero = tmp_path / "fake_libero_client.py"
    libero.write_text(FAKE_LIBERO_CLIENT, encoding="utf-8")
    robotwin = tmp_path / "fake_robotwin_client.py"
    robotwin.write_text(FAKE_ROBOTWIN_CLIENT, encoding="utf-8")
    return {"libero": libero, "robotwin": robotwin}


def _libero_identity(entry: dict) -> str:
    return (
        f"libero:{entry['suite']}:{entry['task_id']}:{entry['initial_state']}:"
        f"{entry['lighting_domain']}"
    )


DOMAINS = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")


def _libero_case(suite: str = "libero_spatial", task_id: int = 1):
    initial_state = 5 * (task_id + 1)
    entries = [
        {
            "platform": "libero",
            "suite": suite,
            "task": f"{suite}:{task_id}",
            "task_id": task_id,
            "initial_state": initial_state,
            "episode": index,
            "env_seed": 3 * task_id,
            "lighting_domain": domain,
            "lighting_ev": -5.0 + index,
            "noise_seed": 1000 + index,
            "benchmark": "rawvla-bench-light-libero-v1",
            "identity": (f"libero:{suite}:{task_id}:{initial_state}:{domain}"),
        }
        for index, domain in enumerate(DOMAINS)
    ]
    official = [
        {
            "suite": suite,
            "task_id": task_id,
            "init_state_index": initial_state,
            "lighting_domain": entry["lighting_domain"],
            "lighting_ev": entry["lighting_ev"],
            "noise_seed": entry["noise_seed"],
            "base_seed": entry["env_seed"],
            "episode_idx": index,
        }
        for index, entry in enumerate(entries)
    ]
    manifest = {
        "benchmark": "rawvla-bench-light-libero-v1",
        "benchmark_seed": 7,
        "entries": official,
    }
    return entries, manifest


def _robotwin_case(task: str = "grab_roller", seed: int = 23):
    entries = [
        {
            "platform": "robotwin",
            "task": task,
            "initial_state": seed,
            "episode": seed,
            "env_seed": seed,
            "lighting_domain": domain,
            "lighting_ev": -4.0 + index,
            "noise_seed": 2000 + index,
            "benchmark": "rawvla-bench-light-robotwin2-v1",
            "identity": f"robotwin:{task}:{seed}:{domain}",
        }
        for index, domain in enumerate(DOMAINS)
    ]
    official = [
        {
            "task_name": task,
            "dataset_seed": seed,
            "dataset_episode_index": seed,
            "lighting_domain": entry["lighting_domain"],
            "lighting_ev": entry["lighting_ev"],
            "noise_seed": entry["noise_seed"],
            "lighting_strategy": "environment_only",
        }
        for entry in entries
    ]
    manifest = {"benchmark": "rawvla-bench-light-robotwin2-v1", "entries": official}
    return entries, manifest


def _spec(tmp_path: Path, port: int, clients: dict[str, Path], **overrides) -> LoopSpec:
    robots = tmp_path / "robotwin_repo"
    (robots / "third_party/RoboTwin").mkdir(parents=True, exist_ok=True)
    template = tmp_path / "deploy_policy.yml"
    template.write_text('host: "127.0.0.1"\nport: 5694\npolicy_name: model2robotwin_interface\n')
    spec = LoopSpec(
        root=tmp_path / "n04-root",
        run_dir=tmp_path / "run",
        checkpoint=tmp_path / "checkpoint-step0000001.pt",
        device="cuda:0",
        sim_device="0",
        seed=101,
        libero_python=Path(sys.executable),
        robotwin_python=Path(sys.executable),
        rawvla_source=robots,
        starvla_source=tmp_path / "starvla",
        robotwin_deploy_template=template,
        robotwin_ckpt=tmp_path / "pi05.safetensors",
        policy_command=None,
        policy_startup_timeout=15,
        plans={"policy_ports": {"libero": port, "robotwin": port}, "scope": ["libero"]},
        client_overrides=dict(clients),
    )
    for name, value in overrides.items():
        setattr(spec, name, value)
    return spec


@pytest.fixture
def service():
    server = DummyPolicyService()
    try:
        yield server
    finally:
        server.close()


def _rows(run_dir: Path) -> list[dict]:
    text = (run_dir / "eval_results.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_eval_list_rejects_duplicate_identity(tmp_path):
    payload = {
        "entries": [
            {
                "platform": "libero",
                "suite": "libero_spatial",
                "task_id": 0,
                "initial_state": 0,
                "lighting_domain": "Low",
                "identity": "libero:libero_spatial:0:0:Low",
            }
        ]
        * 2
    }
    path = tmp_path / "dup.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="重复 episode 身份"):
        load_eval_list(path, per_domain=1)


def test_counts_statuses_progress_and_resume(tmp_path, service, monkeypatch):
    entries, manifest = _libero_case()
    clients = _fake_clients(tmp_path)
    monkeypatch.setenv("FAKE_FAIL_DOMAINS", "Low,Over")
    spec = _spec(tmp_path, service.port, clients)
    spec.plans["full_counts"] = {"libero": {domain: 10 for domain in DOMAINS}}
    (spec.run_dir / "progress.json").parent.mkdir(parents=True, exist_ok=True)
    (spec.run_dir / "progress.json").write_text(
        json.dumps(
            {
                "stage": "S2",
                "platform": "libero",
                "arm": "D",
                "seed": 101,
                "phase": "train",
                "step": 7000,
                "budget_steps": 100000,
                "train_loss": 0.42,
                "gpu_memory_gb": 51.5,
                "artifacts": [],
                "log_tail": [],
            }
        )
    )

    report = run_campaign(spec, entries=entries, manifests={"libero": manifest})
    summary = report["summary"]
    assert summary["libero"] == {
        "completed": 5,
        "planned": 50,
        "successes": 3,
        "failures": 2,
        "errors": 0,
    }
    assert summary["completed"] == 5 and summary["successes"] == 3 and summary["errors"] == 0

    rows = _rows(spec.run_dir)
    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"success", "failure"}
    failed = [row for row in rows if row["status"] == "failure"]
    assert all(row["success"] is False and row["infrastructure_error"] is False for row in failed)
    scored = [row for row in rows if row["lighting_domain"] in {"Low", "Over"}]
    assert all(row["timeout"] for row in scored)  # 6 步 = 协议上界 → 记为超时失败
    required = {
        "identity",
        "platform",
        "task",
        "task_id",
        "initial_state",
        "episode",
        "env_seed",
        "lighting_domain",
        "status",
        "success",
        "infrastructure_error",
        "steps",
        "seconds",
        "video",
        "attempt",
    }
    assert required <= set(rows[0])
    assert all(row["attempt"] == 1 for row in rows)
    assert all(row["video"] for row in rows)
    assert all((spec.run_dir / row["video"]).is_file() for row in rows)
    assert {row["identity"] for row in rows} == {_libero_identity(e) for e in entries}

    progress = json.loads((spec.run_dir / "progress.json").read_text())
    assert set(PROGRESS_FIELDS) <= set(progress)
    assert progress["phase"] == "eval"
    assert progress["step"] == 5 and progress["budget_steps"] == 50
    assert progress["train_loss"] == 0.42 and progress["gpu_memory_gb"] == 51.5  # 训练侧字段保留
    assert progress["eval"]["completed"] == 5 and progress["eval"]["successes"] == 3
    assert progress["eval"]["libero"] == {
        "completed": 5,
        "planned": 50,
        "successes": 3,
        "failures": 2,
        "errors": 0,
    }
    assert not list(spec.run_dir.glob("*.tmp"))

    # 断点续跑：已结算身份不重复执行。
    run_campaign(spec, entries=entries, manifests={"libero": manifest})
    assert len(_rows(spec.run_dir)) == 5

    # --force：重跑并追加新行，最后一行结算。
    run_campaign(spec, entries=entries, manifests={"libero": manifest}, force=True)
    rows = _rows(spec.run_dir)
    assert len(rows) == 10
    replayed = rows[-5:]
    assert [row["attempt"] for row in replayed] == [2] * 5
    # 真重跑而不是空手跳过：新行仍然来自客户端（3 成功 / 2 失败），不是基础设施错误。
    assert sorted(row["status"] for row in replayed) == [
        "failure",
        "failure",
        "success",
        "success",
        "success",
    ]
    assert all(row["video"] is None or (spec.run_dir / row["video"]).is_file() for row in rows)


def test_infrastructure_error_retries_once_and_keeps_rows(tmp_path, service, monkeypatch):
    entries, manifest = _libero_case()
    clients = _fake_clients(tmp_path)
    monkeypatch.setenv("FAKE_LIBERO_MODE", "crash")
    spec = _spec(tmp_path, service.port, clients)
    spec.plans["full_counts"] = {"libero": {domain: 10 for domain in DOMAINS}}

    report = run_campaign(spec, entries=entries, manifests={"libero": manifest})
    summary = report["summary"]
    assert summary["completed"] == 0 and summary["successes"] == 0
    assert summary["libero"]["errors"] == 5
    rows = _rows(spec.run_dir)
    assert len(rows) == 5 * MAX_ATTEMPTS  # 每个身份两次尝试都保留
    assert all(row["status"] == "infrastructure_error" for row in rows)
    assert all(row["success"] is None and row["infrastructure_error"] is True for row in rows)
    assert sorted(row["attempt"] for row in rows) == [1] * 5 + [2] * 5
    assert (spec.run_dir / "attempts/libero/libero_spatial-1/attempt1").is_dir()
    assert (spec.run_dir / "attempts/libero/libero_spatial-1/attempt2").is_dir()

    # 重试预算用完：再次启动不会无限重跑。
    run_campaign(spec, entries=entries, manifests={"libero": manifest})
    assert len(_rows(spec.run_dir)) == 5 * MAX_ATTEMPTS


def test_retry_success_supersedes_infrastructure_error(tmp_path, service, monkeypatch):
    entries, manifest = _libero_case()
    clients = _fake_clients(tmp_path)
    monkeypatch.setenv("FAKE_LIBERO_MODE", "crash_attempt1")
    spec = _spec(tmp_path, service.port, clients)
    spec.plans["full_counts"] = {"libero": {domain: 10 for domain in DOMAINS}}

    report = run_campaign(spec, entries=entries, manifests={"libero": manifest})
    summary = report["summary"]
    assert summary["completed"] == 5 and summary["successes"] == 5 and summary["errors"] == 0
    rows = _rows(spec.run_dir)
    assert len(rows) == 10
    assert sorted(row["attempt"] for row in rows) == [1] * 5 + [2] * 5
    assert all(row["status"] == "success" for row in _latest(rows).values())
    progress = json.loads((spec.run_dir / "progress.json").read_text())
    assert progress["eval"]["libero"]["errors"] == 0  # 补跑成功后不再计入错误


def test_partial_attempt_keeps_success_and_retries_missing(tmp_path, service, monkeypatch):
    entries, manifest = _libero_case()
    clients = _fake_clients(tmp_path)
    monkeypatch.setenv("FAKE_LIBERO_MODE", "partial_attempt1")
    spec = _spec(tmp_path, service.port, clients)
    spec.plans["full_counts"] = {"libero": {domain: 10 for domain in DOMAINS}}

    report = run_campaign(spec, entries=entries, manifests={"libero": manifest})
    rows = _rows(spec.run_dir)
    first = [row for row in rows if row["identity"].endswith(":ExtremeLow")]
    assert len(first) == 1 and first[0]["status"] == "success" and first[0]["attempt"] == 1
    others = [row for row in rows if not row["identity"].endswith(":ExtremeLow")]
    # 4 个缺行身份各保留 1 行基础设施错误 + 1 行补跑结果；已成功身份不再重跑。
    assert sorted(row["attempt"] for row in others) == [1, 1, 1, 1, 2, 2, 2, 2]
    assert all(row["status"] == "infrastructure_error" for row in others if row["attempt"] == 1)
    latest = _latest(rows)
    assert all(row["status"] == "success" for row in latest.values())
    assert report["summary"]["successes"] == 5


def test_robotwin_unit_records_status_and_archives_video(tmp_path, service):
    entries, manifest = _robotwin_case()
    clients = _fake_clients(tmp_path)
    spec = _spec(tmp_path, service.port, clients)
    spec.plans["full_counts"] = {"robotwin": {domain: 10 for domain in DOMAINS}}
    spec.plans["scope"] = ["robotwin"]
    spec.platform = "robotwin"

    report = run_campaign(spec, entries=entries[:1], manifests={"robotwin": manifest})
    assert report["summary"]["robotwin"] == {
        "completed": 1,
        "planned": 50,
        "successes": 1,
        "failures": 0,
        "errors": 0,
    }
    row = _rows(spec.run_dir)[0]
    assert row["status"] == "success" and row["success"] is True
    assert row["steps"] == 9 and row["seconds"] == 2.5
    assert row["video"] == "artifacts/robotwin/robotwin:grab_roller:23:ExtremeLow.mp4"
    assert (spec.run_dir / row["video"]).is_file()


def test_platform_list_and_scope_contract():
    assert PLATFORMS == ("libero", "robotwin")
    assert MAX_ATTEMPTS == 2


def _latest(rows: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["identity"]] = row
    return latest


def test_libero_instrumentation_keeps_protocol_and_is_anchor_checked():
    code = build_libero_client(FAKE_LIBERO_SOURCE)
    compile(code, "libero_client_recording.py", "exec")
    assert "client.noise_rng = np.random.default_rng(policy_seed)" in code
    assert 'os.environ.get("RAWVLA_VIDEO_DIR")' in code
    assert '"steps": policy_steps,' in code
    assert 'f"libero:{args.task_suite}:{task_id}:' in code
    # N04 v1.1 服务持有 episode 内状态：episode 边界显式 reset（仅对声明 n04 的服务发）。
    assert 'payload["episode_id"] = getattr(self, "owac_episode_id", None)' in code
    assert 'client.owac_episode_id = (f"{args.task_suite}:{task_id}:' in code
    assert 'get("n04", {}).get("policy", "")' in code
    # 精确本体历史（最近 proprio_history+1 帧真实观测）与首动作块落盘（配对量化用）。
    assert "client.owac_proprio_frames.append(" in code
    assert 'payload["proprio_history"] = np.stack(frames[-depth:])' in code
    # 前 K 个动作块落盘：spawn 块不含 proprio，配对量化必须从第 2 块起。
    assert "RAWVLA_FIRST_CHUNK_DIR" in code and '"chunks_{_owac_id}.npz"' in code
    assert "RAWVLA_CHUNK_COUNT" in code and "chunk_sources" in code
    # 协议本体（相机/重规划/成功判据）不在插桩范围内：原样保留，只在后面加行。
    step_line = "obs, _, done, _ = env.step(action_plan.popleft().tolist())"
    assert code.count(step_line) == FAKE_LIBERO_SOURCE.count(step_line) == 1
    with pytest.raises(ValueError, match="客户端插桩锚点变化"):
        build_libero_client(FAKE_LIBERO_SOURCE.replace("client.reset()", "client.reset_session()"))


def test_robotwin_instrumentation_adds_optix_and_fixed_seed():
    code = build_robotwin_eval(FAKE_ROBOTWIN_SOURCE)
    compile(code, "robotwin_eval.py", "exec")
    assert 'RAWVLA_ROBOTWIN_DENOISER", "").lower() == "optix"' in code
    assert 'seed_instruction_generators(int(os.environ["RAWVLA_LANGUAGE_SEED"]))' in code
    assert code.count("if eval_seeds:\n                    raise") == 2  # 固定身份不再自动换种子
    assert code.index("def main(usr_args):") > code.index("import os")  # optix 覆盖在 import 之后
    # N04 v1.1 服务持有 episode 内状态：第一次推理前必须显式 reset，且只对声明 n04 的服务发。
    assert code.index("RAWVLA_N04_EPISODE_RESET") > code.index("model = get_model(usr_args)")
    assert 'get("n04", {}).get("policy", "")' in code
    # 缺 v1.0 owac_rawvla_clean.yml 时，用官方 manifest 条目的 RAW 传感器参数喂相机。
    assert 'args["hdr_raw_white_level"] = float(_owac_entry["raw_white_level"])' in code
    assert '("noise_seed", "raw_sensor_noise_seed")' in code
    # 渲染 RAW 帧抽检（p50/p99/max，5 域各 ≥2 帧）用的落盘钩子。
    assert "RAWVLA_RAW_FRAME_DUMP" in code and "percentile" in code
    with pytest.raises(ValueError, match="客户端插桩锚点变化"):
        build_robotwin_eval(
            FAKE_ROBOTWIN_SOURCE.replace("def main(usr_args):", "def main(usr_args) :")
        )


def test_every_unit_runs_in_one_campaign(tmp_path, service, monkeypatch):
    """回归：一次 campaign 必须跑完所有 unit（曾经只剩最后一个 unit 被执行）。"""
    entries_a, manifest_a = _libero_case(task_id=1)
    entries_b, manifest_b = _libero_case(task_id=2)
    clients = _fake_clients(tmp_path)
    spec = _spec(tmp_path, service.port, clients)
    spec.plans["full_counts"] = {"libero": {domain: 10 for domain in DOMAINS}}

    report = run_campaign(
        spec,
        entries=entries_a + entries_b,
        manifests={
            "libero": {
                "benchmark": manifest_a["benchmark"],
                "entries": manifest_a["entries"] + manifest_b["entries"],
            }
        },
    )
    rows = _rows(spec.run_dir)
    assert len(rows) == 10
    assert report["summary"]["completed"] == 10 and report["summary"]["successes"] == 10
    assert {row["initial_state"] for row in rows} == {5 * (1 + 1), 5 * (2 + 1)}
    for task_id in (1, 2):
        assert (
            spec.run_dir / f"attempts/libero/libero_spatial-{task_id}/attempt1/episodes.jsonl"
        ).is_file()


def test_robotwin_unit_runs_every_identity(tmp_path, service):
    """回归：RoboTwin 一个 unit 含同 (task, seed) 的 5 个光照域，必须逐条真跑。"""
    entries, manifest = _robotwin_case()
    clients = _fake_clients(tmp_path)
    spec = _spec(tmp_path, service.port, clients)
    spec.plans["full_counts"] = {"robotwin": {domain: 10 for domain in DOMAINS}}
    spec.plans["scope"] = ["robotwin"]
    spec.platform = "robotwin"

    report = run_campaign(spec, entries=entries, manifests={"robotwin": manifest})
    rows = _rows(spec.run_dir)
    assert len(rows) == 5
    assert report["summary"]["robotwin"] == {
        "completed": 5,
        "planned": 50,
        "successes": 5,
        "failures": 0,
        "errors": 0,
    }
    assert {row["lighting_domain"] for row in rows} == set(DOMAINS)
    assert all(row["video"] for row in rows)
    case_dirs = sorted((spec.run_dir / "attempts/robotwin/grab_roller-23/attempt1").glob("case*"))
    assert len(case_dirs) == 5
