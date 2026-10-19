"""N04 v1.1 S2 每模型 100 次闭环执行器（LIBERO 50 + RoboTwin 50）。

职责：把 ``configs/n04/v1_1/eval100.json`` 的 episode 身份真正跑到官方仿真器上，产出
逐条 ``eval_results.jsonl``、监控页读的 ``progress.json`` 计数和成败视频
``artifacts/<platform>/<identity>.mp4``；返回码、日志与产物都留在 run 目录里可核对。

计数口径（v1.1 方案 §10、接口契约 §7、S2 提示词「每个待评模型 100 次闭环的具体规则」）：

* ``success``／``failure`` 进成功分母；任务没有在协议步数内达成成功判据算 ``failure``，
  并在行里记 ``timeout``；失败行不删；
* ``infrastructure_error`` 单独计数，**不进成功分母、不算失败结论**；同一 identity 换新
  目录最多重试一次（``MAX_ATTEMPTS``），仍失败就如实保留，报告显示「完成 n/planned，
  成功 s/n，基础设施错误 e」；
* 同一 identity 只按 ``eval_results.jsonl`` 的最后一行结算；基础设施错误的补跑成功不会再
  计入错误数；``--force`` 才重跑已结算身份。

断点续跑：每次启动都从 ``eval_results.jsonl`` 恢复，缺行的身份补跑，已结算身份跳过。

视频：LIBERO 客户端本身不录像，本模块在运行目录 ``runtime/`` 生成一份作者客户端的**冻结
副本**并插桩（只写运行目录产物，不改源码）：录像落到 ``artifacts/libero/<identity>.mp4``；
RoboTwin 复用作者 ``eval_video_log`` 的 ffmpeg 录像；拿不到就写 ``null`` 并如实报告。

协议来源（只读，不修改）：LIBERO 用作者
``starVLA/examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py``
（相机 agentview+eye_in_hand、resize_with_pad、状态 8 维、重规划 5 步、成功判据=环境 done）；
RoboTwin 用作者 ``third_party/RoboTwin/script/eval_policy.py`` + starVLA
``Robotwin/eval_files/deploy_policy.yml``（固定 ``ROBOTWIN_EVAL_SEEDS``，重规划与成功判据同作者）。
RoboTwin 的策略客户端固定为本项目 RawVLA-Bench 协议
``owac.runtime.rawvla_robotwin``（固定 flow seed + 光照元数据校验，保住 eval100 身份）；
它连接 18005 的 loopback websocket 策略服务，本执行器只构造命令、解析结果与计数。

CLI 示例（远端）::

    python -m owac.evaluation.n04_v11_closed_loop \
        --config configs/n04/v1_1/runs/libero_D.json \
        --eval-list configs/n04/v1_1/eval100.json \
        --checkpoint <run>/checkpoint-step0001000.pt \
        --output <run> --platform libero --device cuda:0 \
        --policy-command 'export PYTHONPATH=...; <python> -m <n04 policy server> \
            --ckpt {checkpoint} --port {port}'
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_LIGHTING_DOMAINS",
    "MAX_ATTEMPTS",
    "PLATFORMS",
    "LoopSpec",
    "build_libero_client",
    "build_robotwin_eval",
    "identity_of",
    "load_config",
    "load_eval_list",
    "main",
    "run_campaign",
    "summarize_results",
]

PLATFORMS = ("libero", "robotwin")
DEFAULT_LIGHTING_DOMAINS = ("ExtremeLow", "Low", "Normal", "Over", "ExtremeOver")
PER_DOMAIN = 10
#: 同一 identity 允许的执行次数（1 次正常 + 1 次基础设施错误重试）。
MAX_ATTEMPTS = 2
#: 契约 §7 的 progress.json 字段；未产生的值写 null，不用 0 冒充。
PROGRESS_FIELDS = (
    "stage",
    "platform",
    "arm",
    "seed",
    "phase",
    "updated_at",
    "step",
    "budget_steps",
    "train_loss",
    "dev_loss",
    "step_seconds",
    "gpu_memory_gb",
    "eval",
    "artifacts",
    "log_tail",
)
#: 闭环阶段 step/budget_steps 表示「已完成 n / 计划 N」（监控页对 eval 阶段按此解释）。
EVAL_STATUSES = ("success", "failure", "infrastructure_error")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv")

LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
DEFAULT_ROOT = Path("/workspace/owac/research/n04-v1")
DEFAULT_POLICY_PORTS = {"libero": 18000, "robotwin": 18005}
DEFAULT_POLICY_HOST = "127.0.0.1"


# --------------------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------------------


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读 JSONL；忽略空行与未写完的末行（监控页同样的容错口径）。"""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """临时文件 + ``os.replace``：读者不会看到写了一半的 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """读 JSON 配置并展开 ``include``（相对被包含文件所在目录，与训练入口同语义）。"""

    def load_one(target: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
        target = target.resolve()
        if target in seen:
            raise ValueError(f"配置 include 成环：{target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        includes = payload.pop("include", [])
        merged: dict[str, Any] = {}
        for item in includes:
            merged = _deep_merge(merged, load_one(target.parent / item, seen + (target,)))
        return _deep_merge(merged, payload)

    return load_one(Path(path), ())


# --------------------------------------------------------------------------------------
# eval100 清单与官方 manifest 对齐
# --------------------------------------------------------------------------------------


def identity_of(entry: dict[str, Any]) -> str:
    """eval100 的 identity：清单里有就用清单里的，保证逐条对齐。"""
    identity = entry.get("identity")
    if identity:
        return str(identity)
    platform = str(entry["platform"])
    if platform == "libero":
        return (
            f"libero:{entry['suite']}:{entry['task_id']}:"
            f"{entry['initial_state']}:{entry['lighting_domain']}"
        )
    return f"robotwin:{entry['task']}:{entry['env_seed']}:{entry['lighting_domain']}"


def load_eval_list(
    path: str | Path,
    *,
    domains: Sequence[str] = DEFAULT_LIGHTING_DOMAINS,
    per_domain: int = PER_DOMAIN,
) -> dict[str, Any]:
    """读并校验 ``eval100.json``：平台已知、域已知、身份唯一、每平台每域 ``per_domain`` 条。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload["entries"] if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} 不是 eval 清单：期望非空 entries 列表")
    allowed = tuple(domains)
    seen: set[tuple[str, str]] = set()
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        platform = str(entry["platform"])
        if platform not in PLATFORMS:
            raise ValueError(f"未知平台 {platform!r}（记录 {entry}）")
        domain = str(entry["lighting_domain"])
        if domain not in allowed:
            raise ValueError(f"未知光照域 {domain!r}（记录 {entry}）")
        key = (platform, identity_of(entry))
        if key in seen:
            raise ValueError(f"重复 episode 身份 {key}")
        seen.add(key)
        counts.setdefault(platform, {}).setdefault(domain, 0)
        counts[platform][domain] += 1
    for platform in counts:
        for domain in allowed:
            count = counts[platform].get(domain, 0)
            if count != per_domain:
                raise ValueError(f"{platform}/{domain} 有 {count} 个 episode，期望 {per_domain}")
    return {"path": str(path), "entries": entries, "counts": counts, "domains": list(allowed)}


def _manifest_key(platform: str, entry: dict[str, Any]) -> tuple:
    if platform == "libero":
        return (
            str(entry.get("suite")),
            int(entry["task_id"]),
            int(entry["initial_state"]),
            str(entry["lighting_domain"]),
        )
    return (
        str(entry.get("task", entry.get("task_name"))),
        int(entry["env_seed"]),
        str(entry["lighting_domain"]),
    )


def _official_key(platform: str, entry: dict[str, Any]) -> tuple:
    if platform == "libero":
        return (
            str(entry["suite"]),
            int(entry["task_id"]),
            int(entry["init_state_index"]),
            str(entry["lighting_domain"]),
        )
    return (
        str(entry["task_name"]),
        int(entry["dataset_seed"]),
        str(entry["lighting_domain"]),
    )


def join_official(entries: Sequence[dict[str, Any]], manifests: dict[str, dict[str, Any]]):
    """把 eval100 身份对齐到官方 manifest 条目；0 个或多个匹配都直接报错。

    官方条目带 ``base_seed``/``lighting_ev``/``noise_seed``/``lighting_strategy`` 等执行
    参数，这些是不能从 eval100 猜的，必须逐条来自官方清单。
    """
    index: dict[str, dict[tuple, dict[str, Any]]] = {}
    for platform in PLATFORMS:
        manifest = manifests.get(platform)
        if not manifest:
            continue
        table: dict[tuple, dict[str, Any]] = {}
        for item in manifest["entries"]:
            table[_official_key(platform, item)] = item
        index[platform] = table
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in entries:
        platform = str(entry["platform"])
        table = index.get(platform)
        if table is None:
            raise ValueError(f"缺少 {platform} 官方 manifest，无法对齐执行参数")
        matches = table.get(_manifest_key(platform, entry))
        if matches is None:
            raise ValueError(f"官方 manifest 里找不到身份 {identity_of(entry)}")
        joined.append((entry, matches))
    return joined


def build_filtered_manifest(
    source: dict[str, Any], official_entries: Sequence[dict[str, Any]], path: str | Path
) -> Path:
    """按官方 manifest 的顶层结构写子集清单，供作者客户端逐任务加载。"""
    payload = {key: value for key, value in source.items() if key != "entries"}
    payload["entries"] = list(official_entries)
    payload["owac_closed_loop_subset"] = {
        "entries": len(payload["entries"]),
        "generated_at": utcnow(),
        "source_sha256": source.get("sha256"),
    }
    write_json_atomic(path, payload)
    return Path(path)


# --------------------------------------------------------------------------------------
# 作者客户端的冻结副本与插桩（只写运行目录产物，不改源码）
# --------------------------------------------------------------------------------------


def _replace_once(code: str, old: str, new: str, label: str) -> str:
    if code.count(old) != 1:
        raise ValueError(f"客户端插桩锚点变化（{label}）：期望 1 处，实际 {code.count(old)} 处")
    return code.replace(old, new)


_LIBERO_VIDEO_HELPER = '''
def _owac_n04_enabled(metadata, env_name, default="1"):
    """只有声明 n04_v11 的服务才收这些 payload 字段；官方无状态服务不会收到。"""
    if os.environ.get(env_name, default) != "1":
        return False
    return str((metadata or {}).get("n04", {}).get("policy", "")) == "n04_v11"


def _owac_write_video(path, frames):
    """运行目录内的可选录像（imageio-ffmpeg）；失败只报错不改变 episode 结果。"""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=float(os.environ.get("RAWVLA_VIDEO_FPS", "10")),
                     codec="libx264", quality=7)
    return path


'''


def build_libero_client(source_text: str) -> str:
    """给作者 LIBERO 客户端插桩：逐 episode flow 噪声种子、步数／耗时、可选录像。

    只加三件事，不改相机、归一化、重规划或成功判据：
    1. ``client.noise_rng`` 按 identity 派生（同一身份的 5 个光照域共享同一噪声流）；
    2. ledger 行补 ``steps`` / ``seconds`` / ``max_steps`` / ``flow_seed``；
    3. ``RAWVLA_VIDEO_DIR`` 存在时写 ``<identity>.mp4``（agentview 帧，倒置与模型输入一致）。
    """
    code = _replace_once(
        source_text, "import argparse\n", "import argparse\nimport hashlib\n", "hashlib import"
    )
    code = _replace_once(
        code,
        "def run(args: EvalArgs) -> dict[str, Any]:",
        _LIBERO_VIDEO_HELPER + "def run(args: EvalArgs) -> dict[str, Any]:",
        "video helper",
    )
    code = _replace_once(
        code,
        """            action_plan: collections.deque[np.ndarray] = collections.deque()
            client.reset()
""",
        """            action_plan: collections.deque[np.ndarray] = collections.deque()
            client.reset()
            policy_seed_text = (f"{os.environ.get('RAWVLA_FLOW_SEED_SALT', 'owac-n04-v11-loop')}"
                                f"/libero/{args.task_suite}/{task_id}/{init_state_index}")
            policy_seed = int.from_bytes(
                hashlib.sha256(policy_seed_text.encode()).digest()[:4], "little")
            client.noise_rng = np.random.default_rng(policy_seed)
            if rawvla_entry is not None:
                # N04 v1.1 服务持有 episode 内状态：payload 里带 episode_id，服务端在 id
                # 变化时 reset（WS-H 约定）。官方无状态 π0.5 服务不声明 n04，不会带这个键。
                client.owac_episode_id = (f"{args.task_suite}:{task_id}:"
                                          f"{init_state_index}:"
                                          f"{rawvla_entry['lighting_domain']}")
                client.owac_chunks = []
                client.owac_chunk_sources = []
                _owac_depth = int(((client.metadata or {}).get("n04") or {}).get(
                    "proprio_history", 2)) + 1
                client.owac_proprio_frames = collections.deque(maxlen=_owac_depth)
            episode_started = time.perf_counter()
            policy_steps = 0
            owac_frames = []
""",
        "episode seed",
    )
    code = _replace_once(
        code,
        "                example, _ = preprocess_env_obs(obs, task_description,"
        " args.resize_size)\n",
        """                example, _ = preprocess_env_obs(obs, task_description, args.resize_size)
                if _owac_n04_enabled(client.metadata, "RAWVLA_N04_PROPRIO_EXACT"):
                    client.owac_proprio_frames.append(
                        np.asarray(example["raw_state"], dtype=np.float32))
""",
        "N04 proprio frames",
    )
    code = _replace_once(
        code,
        "                obs, _, done, _ = env.step(action_plan.popleft().tolist())\n",
        """                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                policy_steps += 1
                if os.environ.get("RAWVLA_VIDEO_DIR"):
                    owac_frames.append(np.asarray(obs["agentview_image"])[::-1, ::-1].copy())
""",
        "frame capture",
    )
    code = _replace_once(
        code,
        '                    "rawvla_bench_entry": rawvla_entry,\n',
        """                    "rawvla_bench_entry": rawvla_entry,
                    "steps": policy_steps,
                    "seconds": time.perf_counter() - episode_started,
                    "max_steps": int(max_steps),
                    "flow_seed": policy_seed,
""",
        "ledger fields",
    )
    code = _replace_once(
        code,
        '        payload = {"examples": [model_input]}\n',
        """        payload = {"examples": [model_input]}
        if _owac_n04_enabled(self.metadata, "RAWVLA_N04_EPISODE_TAG"):
            payload["episode_id"] = getattr(self, "owac_episode_id", None)
        if _owac_n04_enabled(self.metadata, "RAWVLA_N04_PROPRIO_EXACT"):
            frames = list(getattr(self, "owac_proprio_frames", ()) or ())
            depth = int(((self.metadata or {}).get("n04") or {}).get("proprio_history", 2)) + 1
            while 0 < len(frames) < depth:
                frames.insert(0, frames[0])
            if frames:
                payload["proprio_history"] = np.stack(frames[-depth:]).astype(np.float32)
""",
        "N04 episode_id",
    )
    code = _replace_once(
        code,
        "            model_source=self.model_source,\n        )\n",
        """            model_source=self.model_source,
        )
        _owac_dump_dir = os.environ.get("RAWVLA_FIRST_CHUNK_DIR")
        if _owac_dump_dir:
            # 前 K 个动作块（spawn 块不含 proprio，配对比较要从第 2 块起）。
            _owac_limit = int(os.environ.get("RAWVLA_CHUNK_COUNT", "3"))
            _owac_chunks = getattr(self, "owac_chunks", None)
            if _owac_chunks is None:
                _owac_chunks = self.owac_chunks = []
            if len(_owac_chunks) < _owac_limit:
                _owac_chunks.append(np.asarray(actions, dtype=np.float32))
                _owac_sources = list(getattr(self, "owac_chunk_sources", []))
                _owac_sources.append(
                    str((response.get("data") or {}).get("proprio_history_source", "")))
                self.owac_chunk_sources = _owac_sources
                _owac_id = str(getattr(self, "owac_episode_id", "unknown")).replace("/", "_")
                _owac_path = Path(_owac_dump_dir) / f"chunks_{_owac_id}.npz"
                _owac_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    _owac_path,
                    env_actions=np.stack(_owac_chunks).astype(np.float32),
                    chunk_sources=np.asarray(_owac_sources),
                    episode_id=np.asarray(str(getattr(self, "owac_episode_id", ""))),
                )
""",
        "first chunk dump",
    )
    code = _replace_once(
        code,
        "            task_sr = task_successes / max(task_episodes, 1)\n",
        """            _owac_video_dir = os.environ.get("RAWVLA_VIDEO_DIR")
            if _owac_video_dir and rawvla_entry is not None and owac_frames:
                _owac_identity = (f"libero:{args.task_suite}:{task_id}:"
                                  f"{int(rawvla_entry['init_state_index'])}:"
                                  f"{rawvla_entry['lighting_domain']}")
                _owac_path = Path(_owac_video_dir) / f"{_owac_identity}.mp4"
                _owac_write_video(_owac_path, owac_frames)
                log.info("  wrote video: %s", _owac_path)
            task_sr = task_successes / max(task_episodes, 1)
""",
        "video write",
    )
    return code


_ROBOTWIN_SEED_HELPER = '''def seed_instruction_generators(seed):
    """固定清单种子：同时重置 Python random 与 NumPy（不读未来帧／专家动作）。"""
    import random as _random

    import numpy as _np

    _random.seed(int(seed))
    _np.random.seed(int(seed))


'''


_ROBOTWIN_RESET_BLOCK = """    if os.environ.get("RAWVLA_N04_EPISODE_RESET", "1") == "1":
        # N04 v1.1 服务持有 episode 内状态；RoboTwin 每个 identity 是新进程＋新连接，
        # 第一次推理前显式 reset；官方无状态服务不声明 n04，不会走这里。
        _owac_link = getattr(model, "client", None)
        _owac_metadata = getattr(_owac_link, "_server_metadata", None) or {}
        if str(_owac_metadata.get("n04", {}).get("policy", "")) == "n04_v11":
            from deployment.model_server.tools import msgpack_numpy as _owac_msgpack

            _owac_link._ws.send(_owac_link._packer.pack({"type": "reset"}))
            _owac_reply = _owac_msgpack.unpackb(_owac_link._ws.recv())
            if not _owac_reply.get("ok", False):
                raise RuntimeError(f"N04 策略服务 episode reset 失败: {_owac_reply}")
            print("N04 v1.1 服务 episode reset 完成")


"""


_ROBOTWIN_RAW_FRAME_DUMP = """            _owac_raw_dump = os.environ.get("RAWVLA_RAW_FRAME_DUMP")
            if _owac_raw_dump and TASK_ENV.take_action_cnt < int(
                    os.environ.get("RAWVLA_RAW_FRAME_STEPS", "2")):
                import json as _owac_json

                import numpy as _owac_np

                _owac_key = os.environ.get("ROBOTWIN_IMAGE_OBS_KEY", "raw_linear")
                for _owac_cam in ("head_camera", "left_camera", "right_camera"):
                    _owac_arr = _owac_np.asarray(
                        observation["observation"][_owac_cam][_owac_key], dtype=_owac_np.float32)
                    _owac_p50, _owac_p99 = _owac_np.percentile(_owac_arr, [50, 99]).tolist()
                    with open(_owac_raw_dump, "a", encoding="utf-8") as _owac_stream:
                        _owac_stream.write(_owac_json.dumps({
                            "episode_id": os.environ.get("RAWVLA_CASE_ID", ""),
                            "lighting_domain": os.environ.get("ROBOTWIN_LIGHTING_DOMAIN", ""),
                            "step": int(TASK_ENV.take_action_cnt),
                            "camera": _owac_cam,
                            "key": _owac_key,
                            "dtype": str(_owac_arr.dtype),
                            "shape": list(_owac_arr.shape),
                            "p50": _owac_p50,
                            "p99": _owac_p99,
                            "max": float(_owac_arr.max()),
                            "min": float(_owac_arr.min()),
                            "mean": float(_owac_arr.mean()),
                        }) + "\\n")
"""


_ROBOTWIN_RAW_SENSOR_BLOCK = """    # 无 v1.0 的 owac_rawvla_clean.yml：用官方 manifest 条目里的 RAW
    # 传感器参数（白电平/噪声/饱和/种子）喂作者相机，保持输入域一致。
    _owac_expected = os.environ.get("RAWVLA_EXPECTED_ENTRY")
    if _owac_expected:
        _owac_entry = json.loads(_owac_expected)
        if "raw_white_level" in _owac_entry:
            args["hdr_raw_white_level"] = float(_owac_entry["raw_white_level"])
        for _owac_src, _owac_dst in (
            ("raw_sensor_noise_enabled", "raw_sensor_noise_enabled"),
            ("raw_sensor_shot_noise", "raw_sensor_shot_noise"),
            ("raw_sensor_read_noise", "raw_sensor_read_noise"),
            ("sensor_saturation_ev", "sensor_saturation_ev"),
            ("noise_seed", "raw_sensor_noise_seed"),
            ("extra_light_intensity", "extra_light_intensity"),
        ):
            if _owac_src in _owac_entry:
                args[_owac_dst] = _owac_entry[_owac_src]


"""


_ROBOTWIN_OPTIX_HELPER = """if os.environ.get("RAWVLA_ROBOTWIN_DENOISER", "").lower() == "optix":
    # Blackwell 上 SAPIEN 硬编码的 OIDN 失效；setup_demo 之前在运行时改走 optix。
    try:
        import sapien.render as _owac_sapien_render
    except Exception as _owac_exc:  # pragma: no cover - 环境缺失时如实报错，不静默
        print("RAWVLA_ROBOTWIN_DENOISER=optix but import sapien.render failed:", _owac_exc)
    else:
        _owac_original_denoiser = _owac_sapien_render.set_ray_tracing_denoiser

        def _owac_force_optix(*_owac_args, **_owac_kwargs):
            return _owac_original_denoiser("optix")

        _owac_sapien_render.set_ray_tracing_denoiser = _owac_force_optix


"""


def build_robotwin_eval(source_text: str) -> str:
    """给作者 RoboTwin ``script/eval_policy.py`` 插桩。

    改动：固定清单种子、失败不再自动换种子、Blackwell 上 optix 降噪覆盖。

    与冻结全量评测同样的改动（协议一致性），另外不启用作者的失败重试环：
    固定 manifest 身份必须落在同一 ``dataset_seed`` 上，换种子会改变身份。
    """
    code = source_text
    code = code.replace(
        "            except UnStableError as e:\n",
        "            except UnStableError as e:\n                if eval_seeds:\n"
        "                    raise\n",
    )
    code = code.replace(
        "            except Exception as e:\n",
        "            except Exception as e:\n                if eval_seeds:\n"
        "                    raise\n",
    )
    # 插到作者 import 段之后、main 之前：optix 覆盖必须早于 setup_demo，且此时 os 已导入。
    code = _replace_once(
        code,
        "def main(usr_args):",
        _ROBOTWIN_SEED_HELPER + _ROBOTWIN_OPTIX_HELPER + "def main(usr_args):",
        "main 与 optix 覆盖",
    )
    code = _replace_once(
        code,
        """            observation = TASK_ENV.get_obs()
            # Persist exactly the pre-action observation consumed by the
            # policy, yielding one observation for every predicted action.
""",
        """            observation = TASK_ENV.get_obs()
            # Persist exactly the pre-action observation consumed by the
            # policy, yielding one observation for every predicted action.
"""
        + _ROBOTWIN_RAW_FRAME_DUMP,
        "RAW frame dump",
    )
    code = _replace_once(
        code,
        "    args['task_name'] = task_name\n",
        _ROBOTWIN_RAW_SENSOR_BLOCK + "    args['task_name'] = task_name\n",
        "RAW sensor kwargs",
    )
    code = _replace_once(
        code,
        "    model = get_model(usr_args)\n",
        "    model = get_model(usr_args)\n" + _ROBOTWIN_RESET_BLOCK,
        "N04 episode reset",
    )
    code = _replace_once(
        code,
        "        if use_dataset_instructions:\n",
        '        seed_instruction_generators(int(os.environ["RAWVLA_LANGUAGE_SEED"]))\n'
        "        if use_dataset_instructions:\n",
        "language seed",
    )
    return code


def write_client_copies(spec: LoopSpec, *, runtime_dir: Path) -> dict[str, Path]:
    """把插桩后的客户端写到 run 目录 ``runtime/``，返回可执行路径表。"""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    override = spec.client_overrides.get("libero")
    libero_target = Path(override) if override and Path(override).is_file() else None
    if libero_target is None and spec.libero_client_source and spec.libero_client_source.is_file():
        libero_target = runtime_dir / "libero_client_recording.py"
        libero_target.write_text(
            build_libero_client(spec.libero_client_source.read_text(encoding="utf-8"))
        )
    if libero_target is not None:
        written["libero"] = libero_target
    override = spec.client_overrides.get("robotwin")
    robotwin_target = Path(override) if override and Path(override).is_file() else None
    if (
        robotwin_target is None
        and spec.robotwin_eval_source
        and spec.robotwin_eval_source.is_file()
    ):
        robotwin_target = runtime_dir / "robotwin_eval.py"
        robotwin_target.write_text(
            build_robotwin_eval(spec.robotwin_eval_source.read_text(encoding="utf-8"))
        )
    if robotwin_target is not None:
        written["robotwin"] = robotwin_target
    return written


# --------------------------------------------------------------------------------------
# 运行配置
# --------------------------------------------------------------------------------------


@dataclass
class LoopSpec:
    """一次闭环运行的全部路径与执行参数（默认值面向远端 N04 v1.1 布局）。"""

    root: Path = DEFAULT_ROOT
    run_dir: Path = Path(".")
    checkpoint: Path = Path("")
    data_root: Path | None = None
    device: str = "cuda:0"
    sim_device: str = "0"
    seed: int = 101
    image_representation: str = "default_isp"
    # LIBERO 训练帧 256（服务端 metadata training_obs_image_size 会核对）；作者默认 224。
    resize_size: int = 256
    # 配对实验用：每 episode 首个动作块落盘目录（None 表示不落）。
    first_chunk_dir: Path | None = None
    # 本体历史口径（exact=客户端发最近帧真实观测；approx=服务端按策略调用近似）。
    proprio_mode: str | None = None
    flow_seed_salt: str = "owac-n04-v11-loop"
    record_video: bool = True
    video_fps: int = 10
    starvla_source: Path | None = None
    rawvla_source: Path | None = None
    libero_python: Path | None = None
    robotwin_python: Path | None = None
    libero_client_source: Path | None = None
    robotwin_eval_source: Path | None = None
    libero_manifest: Path | None = None
    robotwin_manifest: Path | None = None
    robotwin_deploy_template: Path | None = None
    # 官方 robotwin manifest 的 task_config 是 demo_clean（含 eval_video_log: true）；
    # 以清单字段为准，这里只是回退默认值。
    # 相机模态：N04 臂 D 训练吃 RAW（fork 的 raw_linear，float32 线性，白电平已除）；
    # 官方 π0.5 对照用 render_rgb（uint8 显示域，v1.0 冻结协议）。
    robotwin_image_obs_key: str = "raw_linear"
    robotwin_task_config: str = "demo_clean"
    robotwin_denoiser: str = "optix"
    # 定案（lead 2026-10-15）：RoboTwin 用本项目固定身份协议 B，保住 eval100 的
    # flow seed／光照域／任务初态；starVLA 的 model2robotwin_interface 是另一套口径。
    robotwin_policy_name: str = "owac.runtime.rawvla_robotwin"
    robotwin_ckpt: Path | None = None
    robotwin_ckpt_setting: str = "owac-n04-v11"
    cuda_home: str = "/usr/local/cuda-12.8"
    vk_icd: str = "/etc/vulkan/icd.d/nvidia_icd.json"
    unit_timeout_seconds: int = 21600
    endpoint_timeout_seconds: int = 900
    policy_host: str = DEFAULT_POLICY_HOST
    policy_command: str | None = None
    policy_startup_timeout: int = 1800
    stage: str = "S2"
    platform: str | None = None
    arm: str = "D"
    plans: dict[str, Any] = field(default_factory=dict)
    #: 已就绪的客户端脚本（跳过插桩）；测试用假客户端或运维预生成客户端时使用。
    client_overrides: dict[str, Path] = field(default_factory=dict)

    def resolve(self) -> LoopSpec:
        root = self.root
        self.starvla_source = self.starvla_source or Path(
            os.environ.get("N04V11_STARVLA", "/workspace/owac/rawvla/source/starVLA")
        )
        rawvla_source = Path(
            os.environ.get("N04V11_RAW_VLA", "/workspace/owac/rawvla/source/RawVLA")
        )
        self.rawvla_source = self.rawvla_source or rawvla_source
        self.libero_python = self.libero_python or Path(
            os.environ.get("N04V11_LIBERO_PYTHON", str(root / "envs/libero-sim/bin/python"))
        )
        self.robotwin_python = self.robotwin_python or Path(
            os.environ.get("N04V11_ROBOTWIN_PYTHON", str(root / "envs/robotwin-sim/bin/python"))
        )
        self.libero_client_source = self.libero_client_source or (
            self.starvla_source
            / "examples/simBenchmarks/LIBERO/eval_files/openpi/eval_starvla_openpi_client.py"
        )
        self.robotwin_eval_source = self.robotwin_eval_source or (
            self.rawvla_source / "third_party/RoboTwin/script/eval_policy.py"
        )
        self.robotwin_deploy_template = self.robotwin_deploy_template or (
            self.starvla_source / "examples/simBenchmarks/Robotwin/eval_files/deploy_policy.yml"
        )
        self.robotwin_ckpt = self.robotwin_ckpt or Path(
            os.environ.get(
                "N04V11_ROBOTWIN_CKPT",
                "/workspace/owac/rawvla/checkpoints/pi05_robotwin/model.safetensors",
            )
        )
        frozen_results = Path(
            os.environ.get(
                "N04V11_FROZEN_RESULTS",
                "/workspace/owac/repo/research/experiments/results/2026-10-10_rawvla_pi05_full_baseline",
            )
        )
        self.libero_manifest = self.libero_manifest or Path(
            os.environ.get("N04V11_LIBERO_MANIFEST", str(frozen_results / "libero-manifest.json"))
        )
        self.robotwin_manifest = self.robotwin_manifest or Path(
            os.environ.get(
                "N04V11_ROBOTWIN_MANIFEST", str(frozen_results / "robotwin-manifest.json")
            )
        )
        return self


# --------------------------------------------------------------------------------------
# 策略服务
# --------------------------------------------------------------------------------------


def _tcp_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class PolicyServer:
    """按 ``policy_command`` 模板起策略服务；退出时确保进程组被关掉。

    模板占位符：``{checkpoint} {port} {host} {device} {platform} {output} {root} {seed}``。
    没有 ``policy_command`` 时不启动（attach 到外部已启动的服务），只做端口等待。
    """

    def __init__(self, spec: LoopSpec, platform: str, log_path: Path):
        self.spec = spec
        self.platform = platform
        self.log_path = log_path
        self.port = (
            self.spec.plans.get("policy_ports", {}).get(platform) or DEFAULT_POLICY_PORTS[platform]
        )
        self.host = spec.policy_host
        self.process: subprocess.Popen | None = None
        self.command: str | None = None

    def render(self) -> str | None:
        if not self.spec.policy_command:
            return None
        return self.spec.policy_command.format(
            checkpoint=str(self.spec.checkpoint),
            port=self.port,
            host=self.host,
            device=self.spec.device,
            platform=self.platform,
            output=str(self.spec.run_dir),
            root=str(self.spec.root),
            seed=self.spec.seed,
        )

    def ensure(self) -> None:
        """确保服务在跑：进程已退出或端口未监听时重启并等待握手端口。

        没有 ``policy_command`` 时进入 attach 模式：只等外部已启动的服务监听端口。
        """
        if self.process is not None and self.process.poll() is None:
            # 已在跑：只按进程存活判断；不再裸 TCP 探测，避免 websocket 服务把它记成坏请求。
            return
        if self.process is not None:
            self.stop()
        self.command = self.render()
        if self.command is None:
            self._wait_ready()
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "command": self.command,
                        "platform": self.platform,
                        "port": self.port,
                        "started_at": utcnow(),
                    }
                )
                + "\n"
            )
            stream.flush()
            self.process = subprocess.Popen(
                ["bash", "-lc", self.command],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._wait_ready(process=self.process)

    def _wait_ready(self, process: subprocess.Popen | None = None) -> None:
        deadline = time.monotonic() + self.spec.policy_startup_timeout
        while time.monotonic() < deadline:
            if _tcp_ready(self.host, self.port):
                return
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"策略服务启动失败 rc={process.returncode}，日志 {self.log_path}"
                )
            time.sleep(2.0)
        raise TimeoutError(
            f"策略服务 {self.host}:{self.port} 在 {self.spec.policy_startup_timeout}s 内未监听"
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()

    def __enter__(self) -> PolicyServer:
        self.ensure()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------------------
# 执行单元
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """一次客户端调用覆盖的 episode 集合。"""

    platform: str
    key: str
    entries: tuple[dict[str, Any], ...]
    official: tuple[dict[str, Any], ...]

    @property
    def identities(self) -> tuple[str, ...]:
        return tuple(identity_of(entry) for entry in self.entries)


def owac_pythonpath() -> list[str]:
    """本包 src 目录（快照内）：项目自带客户端（如 ``owac.runtime.rawvla_robotwin``）需要它。"""
    return [str(Path(__file__).resolve().parents[2])]


def _order_key(entry: dict[str, Any]) -> tuple:
    domain = str(entry["lighting_domain"])
    order = {name: index for index, name in enumerate(DEFAULT_LIGHTING_DOMAINS)}
    if str(entry["platform"]) == "libero":
        return (str(entry.get("suite")), int(entry["task_id"]), order.get(domain, 99))
    return (str(entry.get("task")), int(entry["env_seed"]), order.get(domain, 99))


def plan_units(
    joined: Sequence[tuple[dict[str, Any], dict[str, Any]]], platform: str
) -> list[Unit]:
    """按 (suite, task)／(task, dataset_seed) 分组。

    LIBERO 一次调用跑一个任务的全部选中 episode，RoboTwin 一次一个 episode。
    """
    groups: dict[tuple, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for entry, official in joined:
        if str(entry["platform"]) != platform:
            continue
        key = (
            ("libero", str(entry["suite"]), int(entry["task_id"]))
            if platform == "libero"
            else ("robotwin", str(entry.get("task")), int(entry["env_seed"]))
        )
        groups.setdefault(key, []).append((entry, official))
    units: list[Unit] = []
    for key in sorted(groups, key=lambda item: (item[1], item[2])):
        items = sorted(groups[key], key=lambda pair: _order_key(pair[0]))
        suffix = f"{key[1]}-{key[2]}"
        units.append(
            Unit(
                platform=platform,
                key=suffix,
                entries=tuple(entry for entry, _ in items),
                official=tuple(official for _, official in items),
            )
        )
    return units


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int,
) -> tuple[int, float]:
    """跑一个客户端进程（独立进程组），超时按 124 记录；返回 (returncode, seconds)。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"command": list(command), "cwd": str(cwd), "started_at": utcnow()}) + "\n"
        )
        stream.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
            code = 124
    return code, time.perf_counter() - started


class LiberoRunner:
    """LIBERO 执行单元：一次客户端调用跑一个 (suite, task) 下的全部选中 episode。"""

    platform = "libero"

    def __init__(self, spec: LoopSpec, source_manifest: dict[str, Any], client: Path):
        self.spec = spec
        self.manifest = source_manifest
        self.client = client

    def execute(
        self,
        unit: Unit,
        attempt_dir: Path,
        *,
        host: str,
        port: int,
        only: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        # ``only`` 只被 RoboTwin 逐条路径使用；LIBERO 靠 ledger 续跑跳过已结算 episode。
        _ = only
        attempt_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = build_filtered_manifest(
            self.manifest, unit.official, attempt_dir / "manifest.json"
        )
        ledger = attempt_dir / "episodes.jsonl"
        suite = str(unit.entries[0]["suite"])
        task_id = int(unit.entries[0]["task_id"])
        video_dir = self.spec.run_dir / "artifacts" / "libero"
        video_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.spec.libero_python),
            str(self.client),
            "--model",
            "PI05",
            "--model-source",
            "openpi",
            "--host",
            str(host),
            "--port",
            str(port),
            "--task-suite",
            suite,
            "--task-start",
            str(task_id),
            "--max-tasks",
            "1",
            "--image-mode",
            "rawvla_bench",
            "--rawvla-bench-representation",
            self.spec.image_representation,
            "--rawvla-bench-manifest",
            str(manifest_path),
            "--resume-path",
            str(ledger),
            "--result-json",
            str(attempt_dir / "result.json"),
            "--resize-size",
            str(self.spec.resize_size),
            "--no-log-action-stats",
        ]
        env = os.environ.copy()
        pythonpath = os.pathsep.join(
            [
                str(self.spec.run_dir / "runtime"),
                str(self.spec.starvla_source),
                str(self.spec.starvla_source / "examples/simBenchmarks/LIBERO/eval_files/openpi"),
                str(self.spec.rawvla_source / "experiments/simulation/libero"),
                *owac_pythonpath(),
            ]
        )
        env.update(
            PYTHONPATH=pythonpath,
            LIBERO_CONFIG_PATH=str(self.spec.rawvla_source / ".local/libero"),
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            MUJOCO_EGL_DEVICE_ID=str(self.spec.sim_device),
            TOKENIZERS_PARALLELISM="false",
            RAWVLA_FLOW_SEED_SALT=self.spec.flow_seed_salt,
        )
        if self.spec.first_chunk_dir is not None:
            self.spec.first_chunk_dir.mkdir(parents=True, exist_ok=True)
            env["RAWVLA_FIRST_CHUNK_DIR"] = str(self.spec.first_chunk_dir)
        if self.spec.record_video:
            env["RAWVLA_VIDEO_DIR"] = str(video_dir)
            env["RAWVLA_VIDEO_FPS"] = str(self.spec.video_fps)
        # 看门狗按协议步数给足余量（每步 4s 已是慢策略），超时只判基础设施错误。
        protocol_steps = sum(
            LIBERO_MAX_STEPS.get(str(item["suite"]), 300) + 10 for item in unit.entries
        )
        budget = min(int(self.spec.unit_timeout_seconds), max(900, 600 + 4 * protocol_steps))
        returncode, seconds = _run_process(
            command,
            cwd=attempt_dir,
            env=env,
            log_path=attempt_dir / "run.log",
            timeout=budget,
        )
        return self.collect(unit, ledger, video_dir, returncode, seconds)

    def collect(
        self,
        unit: Unit,
        ledger: Path,
        video_dir: Path,
        returncode: int,
        seconds: float,
    ) -> dict[str, dict[str, Any]]:
        """把 ledger 行按官方条目映射回 identity；缺行由调用方记为基础设施错误。"""
        outcomes: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(ledger):
            official = row.get("rawvla_bench_entry")
            if not isinstance(official, dict):
                continue
            identity = (
                f"libero:{row.get('task_suite')}:{int(row['task_id'])}:"
                f"{int(official['init_state_index'])}:{official['lighting_domain']}"
            )
            video_path = video_dir / f"{identity}.mp4"
            max_steps = row.get("max_steps")
            steps = row.get("steps")
            success = bool(row.get("success"))
            outcomes[identity] = {
                "status": "success" if success else "failure",
                "success": success,
                "timeout": bool(
                    not success
                    and steps is not None
                    and max_steps is not None
                    and steps >= int(max_steps)
                ),
                "steps": steps,
                "seconds": row.get("seconds", seconds),
                "flow_seed": row.get("flow_seed"),
                "video": self._video_field(video_path),
                "returncode": returncode,
                "detail": None if success else "任务未在协议步数内达成成功判据",
            }
        return outcomes

    def _video_field(self, path: Path) -> str | None:
        if not self.spec.record_video or not path.is_file():
            return None
        return str(path.relative_to(self.spec.run_dir))


class RobotwinRunner:
    """RoboTwin 执行单元：每个 identity 一次作者 eval_policy 调用（固定 dataset_seed）。"""

    platform = "robotwin"

    def __init__(self, spec: LoopSpec, source_manifest: dict[str, Any], client: Path):
        self.spec = spec
        self.manifest = source_manifest
        self.client = client

    def _deploy_yml(self, attempt_dir: Path, port: int) -> Path:
        text = self.spec.robotwin_deploy_template.read_text(encoding="utf-8")
        rendered = []
        for line in text.splitlines():
            if line.strip().startswith("port:"):
                line = f"port: {port}"
            rendered.append(line)
        path = attempt_dir / "robotwin_deploy.yml"
        path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
        return path

    def execute(
        self,
        unit: Unit,
        attempt_dir: Path,
        *,
        host: str,
        port: int,
        only: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """每个 identity 一次作者 eval_policy 调用（固定 dataset_seed + 光照域）。

        RoboTwin 的 unit 含同一 (task, dataset_seed) 的 5 个光照域；每个 identity 单独起进程
        （作者 harness 一次只跑一个 case），``only`` 用于跳过本轮已结算的身份。
        """
        self._ensure_task_config_link()
        outcomes: dict[str, dict[str, Any]] = {}
        for index, (entry, official) in enumerate(zip(unit.entries, unit.official, strict=True)):
            identity = identity_of(entry)
            if only is not None and identity not in only:
                continue
            case_dir = attempt_dir / f"case{index:02d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"[robotwin] {identity} -> {case_dir.name} 开始", flush=True)
            outcomes.update(self._run_identity(entry, official, case_dir, host=host, port=port))
        return outcomes

    def _ensure_task_config_link(self) -> None:
        """作者相机加载器按 ``__file__/../task_config`` 找配置。

        冻结副本在 runtime/ 下，所以运行目录需要一个指向作者 task_config 的链接。
        """
        link = self.spec.run_dir / "task_config"
        source = self.spec.rawvla_source / "third_party/RoboTwin/task_config"
        if link.exists() or not source.is_dir():
            return
        try:
            link.symlink_to(source, target_is_directory=True)
            print(f"[robotwin] task_config -> {source}", flush=True)
        except OSError as error:  # pragma: no cover - 已存在/权限异常时如实报错
            print(f"[robotwin] task_config 链接失败：{error}", flush=True)

    def _run_identity(
        self,
        entry: dict[str, Any],
        official: dict[str, Any],
        case_dir: Path,
        *,
        host: str,
        port: int,
    ) -> dict[str, dict[str, Any]]:
        deploy = self._deploy_yml(case_dir, port)
        seed = int(official["dataset_seed"])
        env = os.environ.copy()
        pythonpath = os.pathsep.join(
            [
                str(self.spec.run_dir / "runtime"),
                str(self.spec.starvla_source),
                str(self.spec.rawvla_source / "third_party/RoboTwin"),
                str(self.spec.rawvla_source / "third_party/RoboTwin/script"),
                str(self.spec.starvla_source / "examples/simBenchmarks/Robotwin/eval_files"),
                *owac_pythonpath(),
            ]
        )
        env.update(
            PYTHONPATH=pythonpath,
            PATH=f"{self.spec.robotwin_python.parent}{os.pathsep}{env.get('PATH', '')}",
            CUDA_HOME=self.spec.cuda_home,
            RAWVLA_ROBOTWIN_DENOISER=self.spec.robotwin_denoiser,
            ROBOTWIN_TEST_NUM="1",
            ROBOTWIN_EVAL_SEEDS=str(seed),
            ROBOTWIN_FORCE_EVAL_SEED="1",
            ROBOTWIN_IMAGE_OBS_KEY=self.spec.robotwin_image_obs_key,
            ROBOTWIN_LIGHTING_DOMAIN=str(official["lighting_domain"]),
            ROBOTWIN_LIGHTING_STRATEGY=str(official.get("lighting_strategy", "environment_only")),
            ROBOTWIN_LIGHTING_SAMPLE_UNIT="paired",
            ROBOTWIN_LIGHTING_BENCHMARK_SEED=str(self.manifest.get("benchmark_seed", 0)),
            ROBOTWIN_POLICY_RECORD_EPISODE_DATA="0",
            ROBOTWIN_POLICY_AUDIT_DIR=str(case_dir),
            RAWVLA_STEP_AUDIT=str(case_dir / "actions.jsonl"),
            RAWVLA_RAW_FRAME_DUMP=str(case_dir / "raw_frames.jsonl"),
            RAWVLA_RAW_FRAME_STEPS="2",
            START_EPISODE=str(int(official.get("dataset_episode_index", 0))),
            RAWVLA_FLOW_SEED=str(seed),
            RAWVLA_LANGUAGE_SEED=str(seed),
            RAWVLA_CASE_ID=f"{self.spec.run_dir.name}/{case_dir.name}",
            RAWVLA_EXPECTED_ENTRY=json.dumps(official),
            VK_ICD_FILENAMES=self.spec.vk_icd,
        )
        # 作者 eval_policy.py 只认 --config / --policy_ckpt_path / --overrides REMAINDER：
        # 其余设置必须跟在 --overrides 之后成对给出（与冻结 v1.0 调用一致）。
        command = [
            str(self.spec.robotwin_python),
            str(self.client),
            "--config",
            str(deploy),
            "--policy_ckpt_path",
            str(self.spec.robotwin_ckpt),
            "--overrides",
            "--task_name",
            str(entry.get("task")),
            "--task_config",
            str(self.manifest.get("task_config") or self.spec.robotwin_task_config),
            "--ckpt_setting",
            self.spec.robotwin_ckpt_setting,
            "--policy_name",
            self.spec.robotwin_policy_name,
            "--seed",
            "0",
        ]
        cwd = self.spec.rawvla_source / "third_party/RoboTwin"
        started_at = time.time()
        returncode, seconds = _run_process(
            command,
            cwd=cwd,
            env=env,
            log_path=case_dir / "run.log",
            timeout=int(self.spec.endpoint_timeout_seconds),
        )
        return self.collect(
            identity_of(entry),
            seed,
            case_dir,
            returncode,
            seconds,
            search_root=cwd,
            started_at=started_at,
        )

    def collect(
        self,
        identity: str,
        expected_seed: int,
        case_dir: Path,
        returncode: int,
        seconds: float,
        *,
        search_root: Path | None = None,
        started_at: float = 0.0,
    ) -> dict[str, dict[str, Any]]:
        results = read_jsonl(case_dir / "episode_results.jsonl")
        if returncode != 0 or len(results) != 1:
            return {}
        row = results[0]
        if row.get("eval_seed") is not None and int(row["eval_seed"]) != int(expected_seed):
            raise ValueError(
                f"RoboTwin eval_seed 在评测中变化：{row['eval_seed']} != {expected_seed}"
            )
        success = bool(row.get("success"))
        video_dir = self.spec.run_dir / "artifacts" / "robotwin"
        target = video_dir / f"{identity}.mp4"
        source = self._find_video(case_dir, row, search_root=search_root, started_at=started_at)
        if self.spec.record_video and source is not None and source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        return {
            identity: {
                "status": "success" if success else "failure",
                "success": success,
                "timeout": bool(row.get("timeout", False)),
                "steps": row.get("steps", row.get("num_steps")),
                "seconds": row.get("seconds", seconds),
                "flow_seed": None,
                "video": str(target.relative_to(self.spec.run_dir)) if target.is_file() else None,
                "returncode": returncode,
                "detail": None if success else "任务未在协议步数内达成成功判据",
            }
        }

    @staticmethod
    def _find_video(
        case_dir: Path,
        row: dict[str, Any],
        *,
        search_root: Path | None = None,
        started_at: float = 0.0,
    ) -> Path | None:
        """作者把 ``eval_video_log`` 视频写在 cwd 的 ``eval_result/.../episode*.mp4``。"""
        for candidate in (row.get("video"), row.get("video_path")):
            if candidate:
                path = Path(str(candidate))
                if path.is_file():
                    return path
        matches = sorted(case_dir.rglob("episode*.mp4"))
        if matches:
            return matches[-1]
        if search_root is not None:
            fresh = [
                path
                for path in search_root.glob("eval_result/**/episode*.mp4")
                if path.stat().st_mtime >= started_at
            ]
            if fresh:
                return max(fresh, key=lambda path: path.stat().st_mtime)
        return None


# --------------------------------------------------------------------------------------
# 计数与进度
# --------------------------------------------------------------------------------------


def summarize_results(rows: Iterable[dict[str, Any]], *, planned: dict[str, int]) -> dict[str, Any]:
    """按 identity 的最后一行统计：success/failure 进分母，基础设施错误单独计数。"""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row.get("identity")
        if not identity:
            continue
        latest[str(identity)] = row
    per_platform: dict[str, dict[str, int]] = {
        name: {
            "completed": 0,
            "planned": int(planned.get(name, 0)),
            "successes": 0,
            "failures": 0,
            "errors": 0,
        }
        for name in PLATFORMS
    }
    total = {
        "completed": 0,
        "planned": sum(int(value) for value in planned.values()),
        "successes": 0,
        "failures": 0,
        "errors": 0,
    }
    for row in latest.values():
        platform = str(row.get("platform"))
        bucket = per_platform.setdefault(
            platform,
            {
                "completed": 0,
                "planned": int(planned.get(platform, 0)),
                "successes": 0,
                "failures": 0,
                "errors": 0,
            },
        )
        status = str(row.get("status"))
        if status in ("success", "failure"):
            bucket["completed"] += 1
            total["completed"] += 1
            if status == "success":
                bucket["successes"] += 1
                total["successes"] += 1
            else:
                bucket["failures"] += 1
                total["failures"] += 1
        elif status == "infrastructure_error":
            bucket["errors"] += 1
            total["errors"] += 1
        else:
            raise ValueError(f"结果行状态非法：{status!r}（identity={row.get('identity')}）")
    summary = {
        **{key: total[key] for key in ("completed", "planned", "successes", "failures", "errors")},
        "status": (
            "done"
            if total["planned"] > 0
            and total["completed"] >= total["planned"]
            and total["errors"] == 0
            else ("not_started" if not latest else "running")
        ),
        **per_platform,
    }
    return summary


def build_progress(
    previous: dict[str, Any],
    *,
    summary: dict[str, Any],
    spec: LoopSpec,
    log_tail: Sequence[str],
) -> dict[str, Any]:
    """契约 §7 的 progress.json：保留训练侧字段，只更新评估计数与状态。"""
    payload = dict(previous)
    payload.update(
        {
            "stage": spec.stage or previous.get("stage"),
            "platform": spec.platform or previous.get("platform"),
            "arm": spec.arm or previous.get("arm"),
            "seed": spec.seed if spec.seed is not None else previous.get("seed"),
            "phase": "eval",
            "updated_at": utcnow(),
            "step": int(summary["completed"]),
            "budget_steps": int(summary["planned"]),
            "eval": summary,
            "artifacts": newest_artifacts(spec.run_dir),
            "log_tail": list(log_tail)[-20:],
        }
    )
    for name in ("train_loss", "dev_loss", "step_seconds", "gpu_memory_gb"):
        payload.setdefault(name, None)
    missing = [name for name in PROGRESS_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"progress.json 缺少契约字段：{missing}")
    return payload


def newest_artifacts(run_dir: Path, *, limit: int = 20) -> list[str]:
    root = run_dir / "artifacts"
    if not root.is_dir():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path.relative_to(run_dir)) for path in files[:limit]]


# --------------------------------------------------------------------------------------
# 主驱动
# --------------------------------------------------------------------------------------


def _attempt_number(rows: Sequence[dict[str, Any]], identities: Sequence[str]) -> int:
    wanted = set(identities)
    attempts = [int(row.get("attempt", 1)) for row in rows if row.get("identity") in wanted]
    return (max(attempts) if attempts else 0) + 1


def _latest_by_identity(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("identity"):
            latest[str(row["identity"])] = row
    return latest


def _attempts_left(latest: dict[str, dict[str, Any]], identity: str, *, force: bool) -> int:
    if force or identity not in latest:
        return MAX_ATTEMPTS
    row = latest[identity]
    status = str(row.get("status"))
    if status in ("success", "failure"):
        return 0
    if status == "infrastructure_error":
        return max(0, MAX_ATTEMPTS - int(row.get("attempt", 1)))
    raise ValueError(f"结果行状态非法：{status!r}（identity={identity}）")


def _build_row(
    entry: dict[str, Any],
    *,
    identity: str,
    status: str,
    outcome: dict[str, Any],
    checkpoint: Path,
    attempt: int,
) -> dict[str, Any]:
    success = outcome.get("success")
    if status == "success":
        success = True if success is None else bool(success)
    elif status == "failure":
        success = False if success is None else bool(success)
    else:
        success = None
    return {
        "identity": identity,
        "platform": str(entry["platform"]),
        "task": entry.get("task", entry.get("task_name")),
        "task_id": entry.get("task_id"),
        "initial_state": entry.get("initial_state"),
        "episode": entry.get("episode"),
        "env_seed": entry.get("env_seed"),
        "lighting_domain": entry.get("lighting_domain"),
        "status": status,
        "success": success,
        "infrastructure_error": status == "infrastructure_error",
        "timeout": bool(outcome.get("timeout", False)),
        "steps": outcome.get("steps"),
        "seconds": round(float(outcome.get("seconds") or 0.0), 3),
        "video": outcome.get("video"),
        "attempt": int(attempt),
        "checkpoint": str(checkpoint),
        "flow_seed": outcome.get("flow_seed"),
        "returncode": outcome.get("returncode"),
        "detail": outcome.get("detail"),
        "updated_at": utcnow(),
    }


def run_campaign(
    spec: LoopSpec,
    *,
    entries: Sequence[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    force: bool = False,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """按平台执行闭环；每个 episode 结果立即落盘，progress.json 每行原子更新。"""
    spec = spec.resolve()
    run_dir = spec.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "eval_results.jsonl"
    runtime_dir = run_dir / "runtime"
    log_tail: list[str] = []

    def log(message: str) -> None:
        line = f"{utcnow()} {message}"
        log_tail.append(line)
        if on_log is not None:
            on_log(line)

    clients = write_client_copies(spec, runtime_dir=runtime_dir)
    joined = join_official(entries, manifests)
    # planned 按「本次选中平台 ∪ 结果里已出现的平台」在完整清单上的条数计算：
    # 单平台 run 目录显示 n/50；两个平台写进同一目录时收敛到 n/100。
    full_counts = spec.plans.get("full_counts", {})
    planned = {
        platform: int(sum(full_counts.get(platform, {}).values()))
        for platform in PLATFORMS
        if platform in spec.plans.get("scope", PLATFORMS)
    }
    for entry, _ in joined:
        planned.setdefault(str(entry["platform"]), 0)
    units_by_platform = {platform: plan_units(joined, platform) for platform in PLATFORMS}

    rows = read_jsonl(results_path)
    latest = _latest_by_identity(rows)
    for row in latest.values():
        planned.setdefault(str(row.get("platform")), 0)
    left = {
        identity_of(entry): _attempts_left(latest, identity_of(entry), force=force)
        for entry, _ in joined
    }
    started = time.perf_counter()
    unit_runs: dict[tuple[str, str], int] = {}
    if not any(value > 0 for value in left.values()):
        log("没有待执行身份：已结算或基础设施错误重试预算已用完；重跑用 --force")
    for platform in PLATFORMS:
        units = units_by_platform[platform]
        if not any(left[identity] > 0 for unit in units for identity in unit.identities):
            continue
        if platform not in clients:
            raise SystemExit(f"缺少 {platform} 客户端源码（{platform} 路径未解析），无法执行")
        server = PolicyServer(spec, platform, run_dir / "logs" / f"policy-{platform}.log")
        try:
            # 每轮把所有还有重试预算的 unit 跑一遍；基础设施错误最多再补一轮。
            for _round in range(1, MAX_ATTEMPTS + 1):
                pending_units = [
                    unit
                    for unit in units
                    if any(left[identity] > 0 for identity in unit.identities)
                ]
                if not pending_units:
                    break
                for unit in pending_units:
                    if not any(left[identity] > 0 for identity in unit.identities):
                        continue
                    attempts = _attempt_number(read_jsonl(results_path), unit.identities)
                    attempt_dir = run_dir / "attempts" / platform / unit.key / f"attempt{attempts}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    unit_runs[(platform, unit.key)] = unit_runs.get((platform, unit.key), 0) + 1
                    # --force 的第一次执行必须重跑全部 episode；重试才沿用上一次 ledger。
                    carry = attempts > 1 and (unit_runs[(platform, unit.key)] > 1 or not force)
                    _carry_ledger(run_dir, platform, unit.key, attempts, attempt_dir, carry=carry)
                    server.ensure()
                    log(
                        f"[{platform}] {unit.key} attempt{attempts}: {len(unit.identities)} episode"
                    )
                    if platform == "libero":
                        runner: Any = LiberoRunner(spec, manifests["libero"], clients["libero"])
                    else:
                        runner = RobotwinRunner(spec, manifests["robotwin"], clients["robotwin"])
                    try:
                        pending = {identity for identity in unit.identities if left[identity] > 0}
                        outcomes = runner.execute(
                            unit,
                            attempt_dir,
                            host=spec.policy_host,
                            port=server.port,
                            only=pending,
                        )
                    except Exception as error:  # noqa: BLE001 - 执行器异常是基础设施错误，不吞成成功
                        log(f"[{platform}] {unit.key} 执行器异常：{type(error).__name__}: {error}")
                        outcomes = {}
                    for entry in unit.entries:
                        identity = identity_of(entry)
                        if left[identity] <= 0:
                            continue
                        outcome = outcomes.get(identity)
                        if outcome is None:
                            status = "infrastructure_error"
                            outcome = {
                                "success": None,
                                "seconds": None,
                                "detail": (
                                    "客户端未产出该 episode 的结果行（进程失败／超时／缺结果文件）"
                                ),
                            }
                        else:
                            status = str(outcome["status"])
                        row = _build_row(
                            entry,
                            identity=identity,
                            status=status,
                            outcome=outcome,
                            checkpoint=spec.checkpoint,
                            attempt=attempts,
                        )
                        append_jsonl(results_path, row)
                        left[identity] -= 1
                        if status != "infrastructure_error":
                            left[identity] = 0
                        video_note = "yes" if row["video"] else "no"
                        log(
                            f"[{platform}] {identity} -> {status} "
                            f"steps={row['steps']} {row['seconds']}s video={video_note}"
                        )
                        rows.append(row)
                    write_progress_file(spec, rows, planned=planned, log_tail=log_tail)
        finally:
            server.stop()

    summary_rows = read_jsonl(results_path)
    summary = summarize_results(summary_rows, planned=planned)
    payload = write_progress_file(spec, summary_rows, planned=planned, log_tail=log_tail)
    write_json_atomic(
        run_dir / "eval-summary.json", {**summary, "seconds": time.perf_counter() - started}
    )
    log(
        f"闭环结束：{summary['completed']}/{summary['planned']} 完成，"
        f"成功 {summary['successes']}，失败 {summary['failures']}，基础设施错误 {summary['errors']}"
    )
    return {"summary": summary, "progress": payload}


def _carry_ledger(
    run_dir: Path, platform: str, unit_key: str, attempt: int, attempt_dir: Path, *, carry: bool
) -> None:
    """重试时沿用上一次的 ledger：已结算的 episode 不会被客户端重跑。"""
    if attempt <= 1 or not carry:
        return
    if platform == "libero":
        name = "episodes.jsonl"
    else:
        return
    target = attempt_dir / name
    if target.exists():
        return
    previous = run_dir / "attempts" / platform / unit_key / f"attempt{attempt - 1}" / name
    if previous.is_file():
        shutil.copyfile(previous, target)


def write_progress_file(
    spec: LoopSpec,
    rows: Sequence[dict[str, Any]],
    *,
    planned: dict[str, int],
    log_tail: Sequence[str],
) -> dict[str, Any]:
    progress_path = spec.run_dir / "progress.json"
    previous: dict[str, Any] = {}
    if progress_path.exists():
        try:
            previous = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    summary = summarize_results(rows, planned=planned)
    if spec.proprio_mode:
        # 自描述：这份 progress.json 的正式数字是哪一种本体历史口径（契约允许加字段）。
        summary["proprio_history_mode"] = spec.proprio_mode
    payload = build_progress(previous, summary=summary, spec=spec, log_tail=log_tail)
    write_json_atomic(progress_path, payload)
    return payload


def format_report(summary: dict[str, Any]) -> str:
    """「完成 n/planned，成功 s/n，基础设施错误 e」的人类可读口径。"""
    lines = []
    for platform in PLATFORMS:
        bucket = summary.get(platform, {})
        lines.append(
            f"{platform:<9} 完成 {bucket.get('completed', 0)}/{bucket.get('planned', 0)}，"
            f"成功 {bucket.get('successes', 0)}/{bucket.get('completed', 0)}，"
            f"基础设施错误 {bucket.get('errors', 0)}"
        )
    lines.append(
        f"{'总计':<9} 完成 {summary.get('completed', 0)}/{summary.get('planned', 0)}，"
        f"成功 {summary.get('successes', 0)}/{summary.get('completed', 0)}，"
        f"基础设施错误 {summary.get('errors', 0)}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="N04 v1.1 S2 每模型 100 次闭环执行器（LIBERO 50 + RoboTwin 50）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="N04 v1.1 运行配置（支持 include）")
    parser.add_argument("--data-root", default=None, help="官方数据根目录（覆盖配置 data.root）")
    parser.add_argument(
        "--output", required=True, help="run 目录：写 eval_results.jsonl/progress.json/artifacts"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="策略服务设备，如 cuda:0")
    parser.add_argument(
        "--sim-device", default=None, help="仿真渲染 GPU 序号（MUJOCO_EGL_DEVICE_ID）"
    )
    parser.add_argument("--eval-list", default=None, help="eval100.json；默认取配置 eval.list")
    parser.add_argument(
        "--checkpoint", required=True, help="待评 checkpoint（写进结果行，交给策略服务）"
    )
    parser.add_argument(
        "--platform",
        default="all",
        choices=("all", *PLATFORMS),
        help="只跑某个平台；all 表示清单里出现的全部平台",
    )
    parser.add_argument(
        "--root", default=str(DEFAULT_ROOT), help="远端 N04 根目录（envs/ 与 runs/ 所在）"
    )
    parser.add_argument(
        "--policy-command", default=None, help="策略服务启动模板（{checkpoint} {port} 等占位符）"
    )
    parser.add_argument("--policy-host", default=DEFAULT_POLICY_HOST)
    parser.add_argument("--policy-startup-timeout", type=int, default=1800)
    parser.add_argument(
        "--image-representation", default="default_isp", choices=("default_isp", "raw")
    )
    parser.add_argument(
        "--resize-size", type=int, default=256, help="LIBERO 客户端 resize 尺寸（训练帧 256）"
    )
    parser.add_argument(
        "--robotwin-episode-timeout",
        type=int,
        default=1200,
        help="RoboTwin 单 episode 看门狗秒数（成功判据仍是作者 step_lim，不受此值影响）",
    )
    parser.add_argument(
        "--proprio-mode",
        default=None,
        choices=("exact", "approx"),
        help="本体历史口径标记（写进 progress.json；实际行为由 RAWVLA_N04_PROPRIO_EXACT 决定）",
    )
    parser.add_argument(
        "--first-chunk-dir",
        default=None,
        help="把每个 episode 的首个动作块写到该目录（approx/exact 配对量化用）",
    )
    parser.add_argument(
        "--robotwin-image-key",
        default="raw_linear",
        help="RoboTwin 相机模态（臂 D 用 raw_linear；π0.5 对照用 render_rgb）",
    )
    parser.add_argument("--flow-seed-salt", default="owac-n04-v11-loop")
    parser.add_argument(
        "--limit", type=int, default=0, help="每个平台只跑前 N 条（冒烟用；不改身份顺序）"
    )
    parser.add_argument(
        "--per-domain",
        type=int,
        default=PER_DOMAIN,
        help="清单每平台每域期望条数（eval100 为 10；冒烟子集可调小，正式批次不要改）",
    )
    parser.add_argument("--force", action="store_true", help="重跑已结算身份（默认断点续跑）")
    parser.add_argument("--no-video", action="store_true", help="不录像（视频字段写 null）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不启动仿真")
    return parser


def build_spec(args: argparse.Namespace, config: dict[str, Any]) -> LoopSpec:
    closed = config.get("closed_loop") if isinstance(config.get("closed_loop"), dict) else {}
    device = args.device or config.get("device") or "cuda:0"
    sim_device = args.sim_device
    if sim_device is None:
        sim_device = str(device).split(":")[-1] if ":" in str(device) else "0"
    spec = LoopSpec(
        root=Path(args.root),
        run_dir=Path(args.output),
        checkpoint=Path(args.checkpoint),
        data_root=Path(args.data_root) if args.data_root else None,
        device=str(device),
        sim_device=str(sim_device),
        seed=int(args.seed if args.seed is not None else config.get("seed", 101)),
        image_representation=args.image_representation,
        resize_size=int(args.resize_size),
        first_chunk_dir=Path(args.first_chunk_dir) if args.first_chunk_dir else None,
        proprio_mode=args.proprio_mode,
        robotwin_image_obs_key=str(args.robotwin_image_key),
        flow_seed_salt=args.flow_seed_salt,
        record_video=not args.no_video,
        video_fps=int(closed.get("video_fps", 10)),
        stage=str(config.get("stage", "S2")),
        arm=str(config.get("arm", "D")),
        platform=None if args.platform == "all" else args.platform,
        policy_host=args.policy_host,
        policy_command=args.policy_command or closed.get("policy_command"),
        policy_startup_timeout=int(args.policy_startup_timeout),
        unit_timeout_seconds=int(closed.get("timeout_seconds", 21600)),
        endpoint_timeout_seconds=int(
            closed.get("episode_timeout_seconds") or args.robotwin_episode_timeout
        ),
        plans={"policy_ports": closed.get("policy_ports", {})},
    )
    overrides = {
        "starvla_source": closed.get("starvla_source"),
        "rawvla_source": closed.get("rawvla_source"),
        "libero_python": closed.get("libero_python"),
        "robotwin_python": closed.get("robotwin_python"),
        "libero_client_source": closed.get("libero_client_source"),
        "robotwin_eval_source": closed.get("robotwin_eval_source"),
        "robotwin_deploy_template": closed.get("robotwin_deploy_template"),
        "robotwin_task_config": closed.get("robotwin_task_config"),
        "robotwin_policy_name": closed.get("robotwin_policy_name"),
        "robotwin_ckpt_setting": closed.get("robotwin_ckpt_setting"),
        "robotwin_ckpt": closed.get("robotwin_ckpt"),
        "cuda_home": closed.get("cuda_home"),
        "vk_icd": closed.get("vk_icd"),
        "endpoint_timeout_seconds": closed.get("episode_timeout_seconds"),
    }
    for name, value in overrides.items():
        if value not in (None, ""):
            setattr(
                spec,
                name,
                Path(value)
                if name.endswith(("_source", "_python", "_template", "_ckpt"))
                else value,
            )
    clients = closed.get("clients") or {}
    spec.client_overrides = {
        str(platform): Path(value) for platform, value in clients.items() if value
    }
    return spec.resolve()


def _select_entries(
    entries: Sequence[dict[str, Any]], *, platform: str, limit: int
) -> list[dict[str, Any]]:
    selected = [
        entry for entry in entries if platform == "all" or str(entry["platform"]) == platform
    ]
    if limit > 0:
        kept: dict[str, int] = {}
        limited: list[dict[str, Any]] = []
        for entry in selected:
            name = str(entry["platform"])
            kept[name] = kept.get(name, 0) + 1
            if kept[name] <= limit:
                limited.append(entry)
        selected = limited
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.data_root:
        config.setdefault("data", {})["root"] = args.data_root
    eval_list = args.eval_list or (config.get("eval") or {}).get("list")
    if not eval_list:
        raise SystemExit("需要 --eval-list 或配置 eval.list 指向 eval100.json")
    spec = build_spec(args, config)
    domains = tuple((config.get("eval") or {}).get("lighting_domains", DEFAULT_LIGHTING_DOMAINS))
    listing = load_eval_list(eval_list, domains=domains, per_domain=args.per_domain)
    entries = _select_entries(listing["entries"], platform=args.platform, limit=args.limit)
    selected_platforms = [
        platform for platform in PLATFORMS if any(str(e["platform"]) == platform for e in entries)
    ]
    closed = (
        (config.get("closed_loop") or {}) if isinstance(config.get("closed_loop"), dict) else {}
    )
    manifest_paths = closed.get("manifests", {})
    manifests: dict[str, dict[str, Any]] = {}
    for platform in selected_platforms:
        path = manifest_paths.get(platform) or getattr(spec, f"{platform}_manifest")
        if not path or not Path(path).is_file():
            raise SystemExit(
                f"缺少 {platform} 官方 manifest（{path}）："
                f"请用 closed_loop.manifests.{platform} 指定"
            )
        manifests[platform] = json.loads(Path(path).read_text(encoding="utf-8"))
    spec.plans["full_counts"] = listing["counts"]
    spec.plans["scope"] = list(selected_platforms)
    if len(selected_platforms) == 1:
        spec.platform = selected_platforms[0]
    if args.dry_run:
        joined = join_official(entries, manifests)
        units = {platform: plan_units(joined, platform) for platform in PLATFORMS}
        print(
            json.dumps(
                {
                    "output": str(spec.run_dir),
                    "checkpoint": str(spec.checkpoint),
                    # planned_episodes 是 progress.json 的分母：来自完整清单，不受 --limit 影响。
                    "planned_episodes": {
                        platform: int(sum(listing["counts"].get(platform, {}).values()))
                        for platform in selected_platforms
                    },
                    "selected_episodes": {
                        platform: sum(len(u.entries) for u in units[platform])
                        for platform in PLATFORMS
                    },
                    "units": {platform: [u.key for u in units[platform]] for platform in PLATFORMS},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    # 逐条进度打到 stdout：`n04_v11_loop.sh` 的 tee 会把它落到 <run>/logs/loop-*.log。
    report = run_campaign(
        spec, entries=entries, manifests=manifests, force=args.force, on_log=print
    )
    print(format_report(report["summary"]))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
