"""All-RawVLA-Bench RAW loader and tokens-only teacher cache.

This is a thin data path around the existing visual student/teacher.  It does
not train an action or language backend and never reads paired RGB.
"""

from __future__ import annotations

import json
import os
import random
import struct
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

DOMAIN_CAMERAS = {
    "libero": ("base_0_rgb", "left_wrist_0_rgb"),
    "robotwin2": ("robotwin_head", "robotwin_left", "robotwin_right"),
}
DOMAIN_RAW_KEYS = {
    "libero": ("agentview_raw_uint8", "wrist_raw_uint8"),
    "robotwin2": (
        "head_camera_raw_float32",
        "left_camera_raw_float32",
        "right_camera_raw_float32",
    ),
}
IMAGE_SIZE = 256
HISTORY = 6
STRIDE = 5


class AllBenchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BenchSource:
    group_id: str
    domain: str
    suite: str
    task_name: str
    split: str
    path: Path
    num_frames: int
    lighting: str
    cameras: tuple[str, ...]

    @property
    def endpoints(self) -> tuple[int, ...]:
        return tuple(range(0, self.num_frames, STRIDE))

    @property
    def train_endpoints(self) -> tuple[int, ...]:
        return tuple(e for e in self.endpoints if e >= 25)


def load_inventory(path: str | Path) -> tuple[list[BenchSource], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise AllBenchError("inventory has no groups")
    sources = []
    for group in groups:
        domain = str(group["domain"])
        rel = str(group["relpath"])
        # inventory path is relative to the shared raw root
        candidates = []
        if group.get("path"):
            candidates.append(Path(str(group["path"])))
        candidates.append(Path("/root/autodl-pub/datasets/rawvla") / "data" / domain / "raw" / rel)
        path_value = candidates[-1]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    path_value = candidate
                    break
            except PermissionError:
                continue
        sources.append(
            BenchSource(
                group_id=str(group["group_id"]),
                domain=domain,
                suite=str(group["suite"]),
                task_name=str(group.get("task_name", "")),
                split=str(group["split"]),
                path=path_value,
                num_frames=int(group["num_frames"]),
                lighting=str(group["lighting"]),
                cameras=DOMAIN_CAMERAS[domain],
            )
        )
    return sources, payload


def _resize_full_fov(values: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(values)).permute(0, 3, 1, 2).float()
    return F.interpolate(
        tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )


def normalize_raw(values: np.ndarray, domain: str) -> torch.Tensor:
    """Normalize without changing physical RAW semantics."""
    if domain == "libero":
        if values.dtype != np.uint8:
            raise AllBenchError(f"LIBERO RAW must be uint8, got {values.dtype}")
        return torch.from_numpy(values).permute(0, 3, 1, 2).float() / 255.0
    if domain == "robotwin2":
        if values.dtype != np.float32:
            raise AllBenchError(f"RoboTwin RAW must be float32, got {values.dtype}")
        if values.ndim != 4 or values.shape[1:] != (240, 320, 3):
            raise AllBenchError(f"RoboTwin RAW shape differs: {values.shape}")
        return _resize_full_fov(values)
    raise AllBenchError(f"unsupported domain {domain!r}")


def read_npz_member(path: str | Path, member: str) -> Any:
    """Zero-copy readonly memmap for a numeric NPY member stored uncompressed.

    The local ZIP header is parsed only; no extraction, CRC scan, hashing or
    dataset write. Compressed/non-stored members fall back to normal np.load.
    """
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            with np.load(archive_path, allow_pickle=False) as loaded:
                return loaded[member]
        header_offset = int(info.header_offset)
    with archive_path.open("rb") as raw:
        raw.seek(header_offset)
        header = raw.read(30)
        if len(header) != 30:
            raise AllBenchError(f"truncated local ZIP header: {archive_path}")
        signature, _ver, _flags, method, _t, _d, _crc, _cs, _us, fn_len, extra_len = struct.unpack(
            "<4s5H3I2H", header
        )
        if signature != b"PK\x03\x04" or method != zipfile.ZIP_STORED:
            with np.load(archive_path, allow_pickle=False) as loaded:
                return loaded[member]
        raw.seek(header_offset + 30 + int(fn_len) + int(extra_len))
        version = np.lib.format.read_magic(raw)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(raw)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(raw)
        else:
            with np.load(archive_path, allow_pickle=False) as loaded:
                return loaded[member]
        data_offset = raw.tell()
    if dtype.hasobject:
        with np.load(archive_path, allow_pickle=False) as loaded:
            return loaded[member]
    return np.memmap(
        archive_path,
        mode="r",
        offset=data_offset,
        dtype=dtype,
        shape=shape,
        order="F" if fortran else "C",
    )


class Episode:
    def __init__(self, source: BenchSource) -> None:
        self.source = source
        self.raw = {
            key: read_npz_member(source.path, f"{key}.npy")
            for key in DOMAIN_RAW_KEYS[source.domain]
        }
        self.frames = int(next(iter(self.raw.values())).shape[0])
        if self.frames != source.num_frames:
            raise AllBenchError(f"frame count differs for {source.group_id}")

    def burst(self, endpoint: int) -> torch.Tensor:
        from owac.baselines.raw_temporal_siglip import causal_frame_ids

        ids = causal_frame_ids(endpoint)
        values = [
            normalize_raw(np.asarray(self.raw[key][list(ids)]), self.source.domain)
            for key in DOMAIN_RAW_KEYS[self.source.domain]
        ]
        return torch.stack(values, dim=0)  # [Ncam, T, 3, 256, 256]


class EpisodeCache:
    def __init__(self, sources: Mapping[str, BenchSource], capacity: int = 256) -> None:
        import threading

        self.sources = dict(sources)
        self.capacity = int(capacity)
        self._items: dict[str, Episode] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._executor = None
        self._futures: dict[str, Any] = {}

    def _load(self, source_id: str) -> Episode:
        with self._lock:
            if source_id in self._items:
                return self._items[source_id]
        item = Episode(self.sources[source_id])
        with self._lock:
            if source_id not in self._items:
                if len(self._items) >= self.capacity:
                    old = self._order.pop(0)
                    self._items.pop(old, None)
                self._items[source_id] = item
                self._order.append(source_id)
            return self._items[source_id]

    def prefetch(self, source_ids: Sequence[str], workers: int = 4) -> None:
        from concurrent.futures import ThreadPoolExecutor

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        for source_id in source_ids:
            with self._lock:
                if source_id in self._items or source_id in self._futures:
                    continue
            self._futures[source_id] = self._executor.submit(self._load, source_id)

    def get(self, source_id: str) -> Episode:
        with self._lock:
            future = self._futures.pop(source_id, None)
        if future is not None:
            return future.result()
        return self._load(source_id)


class TargetStore:
    """Per-source tokens-only shard: .npy [windows, Ncam, 256, 2048] FP32 + JSON."""

    def __init__(self, root: str | Path, *, max_bytes: int | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.used_bytes = 0
        self._meta_cache: dict[str, dict[str, Any]] = {}
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    self.used_bytes += path.stat().st_size
            except OSError:
                continue

    def reserve(self, byte_count: int) -> None:
        if self.max_bytes is not None and self.used_bytes + int(byte_count) > self.max_bytes:
            raise AllBenchError(
                f"disk cap exceeded: used={self.used_bytes} new={byte_count} cap={self.max_bytes}"
            )
        self.used_bytes += int(byte_count)

    def paths(self, source_id: str) -> tuple[Path, Path]:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in source_id)
        return self.root / f"{safe}.npy", self.root / f"{safe}.json"

    def exists(self, source_id: str) -> bool:
        npy, meta = self.paths(source_id)
        if not npy.is_file() or not meta.is_file():
            return False
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            return bool(info.get("complete")) and info.get("windows") == int(
                np.load(npy, mmap_mode="r").shape[0]
            )
        except Exception:
            return False

    def discard_partial(self, source_id: str) -> None:
        for path in self.paths(source_id):
            path.unlink(missing_ok=True)

    def read(self, source_id: str, endpoint: int) -> Any:
        npy, meta = self.paths(source_id)
        info = self._meta_cache.get(source_id)
        if info is None:
            info = json.loads(meta.read_text(encoding="utf-8"))
            self._meta_cache[source_id] = info
        endpoints = [int(x) for x in info["endpoints"]]
        index = endpoints.index(int(endpoint))
        # Random source locality: own the row then close immediately. OS page
        # cache provides reuse; no per-source mmap FD is retained.
        data = np.load(npy, mmap_mode="r")
        try:
            owned = np.array(data[index], copy=True, dtype=np.float32)
        finally:
            mapping = getattr(data, "_mmap", None)
            if mapping is not None:
                mapping.close()
        tokens = torch.from_numpy(owned)
        mask = torch.ones(tokens.shape[0], dtype=torch.bool)
        return SimpleNamespace(
            teacher_tokens=tokens,
            teacher_token_mask=mask,
            source_id=source_id,
            endpoint=int(endpoint),
        )


def prepare_source(teacher: Any, source: BenchSource, store: TargetStore) -> dict[str, Any]:
    if store.exists(source.group_id):
        info = json.loads(store.paths(source.group_id)[1].read_text(encoding="utf-8"))
        return {**info, "skipped_complete": True}
    store.discard_partial(source.group_id)
    episode = Episode(source)
    endpoints = source.endpoints
    n_cam = len(DOMAIN_CAMERAS[source.domain])
    npy_path, meta_path = store.paths(source.group_id)
    store.reserve(len(endpoints) * n_cam * 256 * 2048 * 4)
    # Preallocate only this source shard; no full-dataset materialization.
    mmap = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(len(endpoints), n_cam, 256, 2048)
    )
    states = {camera: None for camera in DOMAIN_CAMERAS[source.domain]}
    thetas = {camera: None for camera in DOMAIN_CAMERAS[source.domain]}
    device = next(teacher.frontend.parameters()).device
    for row, endpoint in enumerate(endpoints):
        burst = episode.burst(endpoint).to(device)  # [Ncam,T,3,256,256]
        camera_tokens = []
        for camera_index, camera in enumerate(DOMAIN_CAMERAS[source.domain]):
            raw = burst[camera_index : camera_index + 1]
            with torch.no_grad():
                out = teacher.frontend.model(raw, state=states[camera], theta_prev=thetas[camera])
                states[camera] = out.state.detach()
                thetas[camera] = out.theta.detach()
                token = (
                    teacher_embed_rgb(teacher, out.rgb.detach()[0].float().clamp(0, 1).unsqueeze(0))
                    .detach()
                    .float()
                )
            camera_tokens.append(token.reshape(256, 2048))
        mmap[row] = torch.stack(camera_tokens, dim=0).cpu().numpy()
    mmap.flush()
    del mmap
    info = {
        "schema": "owac.all-bench-tokens.v1",
        "source_id": source.group_id,
        "domain": source.domain,
        "cameras": list(DOMAIN_CAMERAS[source.domain]),
        "endpoints": list(endpoints),
        "windows": len(endpoints),
        "frames": source.num_frames,
        "shape": [len(endpoints), n_cam, 256, 2048],
        "dtype": "float32",
        "complete": True,
        "preprocess": "libero_uint8_div255;robotwin_float32_fullfov_bilinear_256",
    }
    temporary = meta_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(info, indent=2), encoding="utf-8")
    os.replace(temporary, meta_path)
    return info


def build_all_bench_student(config: Mapping[str, Any], *, device: str = "cuda:0") -> Any:
    """Build the same full visual student without inherited-tensor audit loops."""
    from owac.baselines.raw_stem_siglip import (
        InitializationAudit,
        LocalConvStem,
        RawStemSiglipStudent,
        VisionGeometry,
    )
    from owac.baselines.raw_temporal_siglip import TemporalRawStemStudent
    from owac.runtime.raw_full_visual_distillation import _vision_architecture

    tower, projector = _native_vision(config, device="cpu")
    arch = _vision_architecture(config)
    student_cfg = dict(config.get("student") or {})
    geometry = VisionGeometry.from_modules(tower, projector)
    patch_stride = geometry.packed_patch_stride(str(student_cfg.get("input_mode", "rawvla_rgb3")))
    stem = LocalConvStem(
        int(arch["num_channels"]),
        int(arch["hidden_size"]),
        patch_stride,
        conv_channels=int(student_cfg.get("stem_hidden", 64)),
        seed=int(student_cfg.get("stem_seed", 101)),
        dtype=torch.float32,
        device="cpu",
    )
    tower.vision_model.embeddings.patch_embedding = stem
    base = RawStemSiglipStudent(
        vision_tower=tower,
        projector=projector,
        input_mode=str(student_cfg.get("input_mode", "rawvla_rgb3")),
        evidence_level="standalone_visual_subset",
        checkpoint=None,
        load_audit=None,
        initialization_audit=InitializationAudit(
            source_tensors=(),
            student_tensors=(),
            stem_tensors=tuple(n for n, _ in stem.named_parameters()),
            checkpoint_sha256="",
            load_audit_sha256="",
            evidence_level="standalone_visual_subset",
        ),
        unfreeze_input_blocks=0,
        allow_input_block_unfreeze=False,
    )
    student = TemporalRawStemStudent(base)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    return student.to(device)


def _native_vision(config: Mapping[str, Any], *, device: str) -> tuple[Any, Any]:
    """Native SigLIP tower + projector with original RGB patch stem."""

    import sys

    from owac.runtime.raw_full_visual_distillation import (
        _vision_architecture,
        load_safetensors_subset,
        parse_safetensors_header,
    )

    assets = config["assets"]
    starvla_root = Path(assets["starvla_root"])
    if str(starvla_root) not in sys.path:
        sys.path.insert(0, str(starvla_root))
    from starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip import (
        SiglipVisionModel,
    )
    from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIMultiModalProjector
    from transformers.models.auto import CONFIG_MAPPING

    arch = _vision_architecture(config)
    hf_config = CONFIG_MAPPING["paligemma"]()
    cfg = hf_config.vision_config
    for name, value in arch.items():
        if name != "projection_dim":
            setattr(cfg, name, value)
    cfg.projection_dim = arch["projection_dim"]
    cfg.projector_hidden_act = "gelu_fast"
    cfg.torch_dtype = "float32"
    cfg._attn_implementation = "eager"
    cfg.hidden_act = "gelu_pytorch_tanh"
    cfg.layer_norm_eps = 1e-6
    tower = SiglipVisionModel(cfg).to(device=device, dtype=torch.float32)
    tower.vision_model.config._attn_implementation = "eager"
    projector = OpenPIMultiModalProjector(hf_config).to(device=device, dtype=torch.float32)
    checkpoint = Path(assets["checkpoint"])
    header, _ = parse_safetensors_header(checkpoint)
    vision_prefix = "vlm.model.model.vision_tower."
    projector_prefix = "vlm.model.model.multi_modal_projector."
    vision_keys = {k: k[len(vision_prefix) :] for k in header if k.startswith(vision_prefix)}
    projector_keys = {
        k: k[len(projector_prefix) :] for k in header if k.startswith(projector_prefix)
    }
    tower.load_state_dict(
        load_safetensors_subset(checkpoint, key_map=vision_keys, dtype=torch.float32), strict=True
    )
    projector.load_state_dict(
        load_safetensors_subset(checkpoint, key_map=projector_keys, dtype=torch.float32),
        strict=True,
    )
    tower.eval()
    projector.eval()
    for parameter in tower.parameters():
        parameter.requires_grad_(False)
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    return tower, projector


def build_vision_teacher(config: Mapping[str, Any], *, device: str = "cuda:0") -> Any:
    """Author ISP + native π0.5 RGB visual tower/projector; no VLA backend."""
    import sys

    assets = config["assets"]
    tower, projector = _native_vision(config, device=device)
    author_root = Path(assets["author_root"])
    starvla_root = Path(assets["starvla_root"])
    for path in (author_root, starvla_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from omegaconf import OmegaConf
    from starVLA.config_loader import load_config
    from starVLA.model.modules.raw_frontend import build_raw_frontend

    cfg = load_config(Path(assets["teacher_config_paths"][-1]))
    raw_cfg = OmegaConf.to_container(cfg.framework.raw_frontend, resolve=True)
    frontend = build_raw_frontend(OmegaConf.create(raw_cfg))
    artifact = torch.load(
        Path(assets["teacher_frontend_checkpoint"]), map_location="cpu", weights_only=True
    )
    state = (
        artifact["frontend"] if isinstance(artifact, dict) and "frontend" in artifact else artifact
    )
    frontend.load_state_dict(state, strict=True)
    frontend.to(device=device, dtype=torch.float32).eval()
    for parameter in frontend.parameters():
        parameter.requires_grad_(False)
    return SimpleNamespace(
        frontend=frontend, model=frontend, vision_tower=tower, projector=projector, device=device
    )


def teacher_embed_rgb(teacher: Any, rgb: torch.Tensor) -> torch.Tensor:
    """Exact native RGB patch/embedding path used by the teacher vision tower."""
    with torch.no_grad():
        image = rgb.float()
        if tuple(image.shape[-2:]) != (224, 224):
            image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = image * 2.0 - 1.0
        hidden = teacher.vision_tower(pixel_values=image).last_hidden_state
        return teacher.projector(hidden)


class AllBenchDataset:
    """Lazy anchors over source-sharded teacher tokens and bounded RAW episodes."""

    def __init__(
        self, sources: Sequence[BenchSource], *, targets: TargetStore, capacity: int = 256
    ) -> None:
        self.sources = {s.group_id: s for s in sources}
        self.targets = targets
        self.cache = EpisodeCache(self.sources, capacity=capacity)

    def packet_for_endpoint(
        self, source_id: str, endpoint: int, device: torch.device | None = None
    ) -> Any:
        from owac.baselines.raw_stem_siglip import RawStemInput
        from owac.baselines.raw_temporal_siglip import (
            TemporalPacket,
            causal_frame_ids,
            causal_padding_mask,
        )

        source = self.sources[source_id]
        burst = self.cache.get(source_id).burst(endpoint)  # [Ncam,6,3,256,256]
        frames = []
        for i in range(HISTORY):
            frame = F.interpolate(
                burst[:, i], size=(224, 224), mode="bilinear", align_corners=False
            )  # calibrated RAW; no finite/minmax/clamp rescan
            frames.append(
                RawStemInput(
                    values=frame.unsqueeze(0).to(device).contiguous()
                    if device is not None
                    else frame.unsqueeze(0).contiguous(),
                    input_mode="rawvla_rgb3",
                    camera_ids=source.cameras,
                    units="normalized_0_1",
                )
            )
        return TemporalPacket(
            frames=frames,
            source_demo_id=source_id,
            endpoint=endpoint,
            frame_ids=causal_frame_ids(endpoint),
            padding_mask=causal_padding_mask(endpoint),
            reset=endpoint == 0,
            timestamps=None,
            position_source_size=256,
            position_target_size=224,
            position_backmap=True,
            preprocess_id="all_bench_fullfov_256_to_224",
        )

    def sample(self, source_id: str, endpoint: int, device: torch.device | None = None) -> Any:

        unroll = tuple(endpoint - 25 + 5 * i for i in range(HISTORY))
        prefix = tuple(range(0, unroll[0], 5))
        target = self.targets.read(source_id, endpoint)

        if device is not None:
            target.teacher_tokens = target.teacher_tokens.to(device)
            target.teacher_token_mask = target.teacher_token_mask.to(device)
        return SimpleNamespace(
            source_id=source_id,
            endpoint=endpoint,
            unroll_packets=tuple(self.packet_for_endpoint(source_id, e, device) for e in unroll),
            prefix_packets=tuple(self.packet_for_endpoint(source_id, e, device) for e in prefix),
            target=target,
            unroll_rows=(),
            charged_context_frames=HISTORY * HISTORY,
            charged_prefix_endpoints=len(prefix),
        )

    def anchor_keys(self, split: str = "train") -> list[tuple[str, int]]:
        keys = []
        for source in self.sources.values():
            if source.split != split:
                continue
            if not self.targets.exists(source.group_id):
                raise FileNotFoundError(f"train target shard missing: {source.group_id}")
            keys.extend((source.group_id, endpoint) for endpoint in source.train_endpoints)
        return keys


def _chunks_for_epoch(
    keys: Sequence[tuple[str, int]], sources: Mapping[str, BenchSource], *, seed: int, epoch: int
) -> list[list[tuple[str, int]]]:

    by_domain: dict[str, list[tuple[str, int]]] = {}
    for key in keys:
        by_domain.setdefault(sources[key[0]].domain, []).append(key)
    chunks: list[list[tuple[str, int]]] = []
    rng = random.Random(seed + epoch)
    for domain in sorted(by_domain):
        items = list(by_domain[domain])
        rng.shuffle(items)
        for start in range(0, len(items), 16):
            chunk = items[start : start + 16]
            # Minimal documented tail padding: repeat the last item.
            while len(chunk) < 16:
                chunk.append(chunk[-1])
            chunks.append(chunk)
    rng.shuffle(chunks)
    return chunks


def validate_all_bench(
    student: Any, dataset: AllBenchDataset, *, device: torch.device, rank: int = 0, world: int = 1
) -> dict[str, Any]:
    """Streaming BF16 validation sharded over ranks; startup/eligible separated."""
    from owac.runtime.raw_full_visual_distillation import set_full_visual_train_mode
    from owac.training.raw_distillation import token_distillation_loss

    rows: list[dict[str, Any]] = []
    original = student.training
    set_full_visual_train_mode(student, False)
    sources = [s for s in dataset.sources.values() if s.split == "validation"]
    try:
        with torch.inference_mode():
            for index, source in enumerate(sources):
                if index % int(world) != int(rank):
                    continue
                state = None
                for endpoint in source.endpoints:
                    packet = dataset.packet_for_endpoint(source.group_id, endpoint)
                    from dataclasses import replace as _replace

                    packet = _replace(
                        packet,
                        frames=tuple(
                            _replace(frame, values=frame.values.to(device))
                            for frame in packet.frames
                        ),
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output, state = student(packet, state)
                    target = dataset.targets.read(source.group_id, endpoint)
                    values = output.projected_tokens.float()
                    teacher = target.teacher_tokens.reshape(
                        1, -1, target.teacher_tokens.shape[-1]
                    ).to(values.device)
                    mask = output.tokens.mask
                    components = token_distillation_loss(values, teacher, mask)
                    mse, cosine, total = components.mse, components.cosine, components.total
                    valid, width = components.valid_tokens, components.width
                    rows.append(
                        {
                            "domain": source.domain,
                            "lighting": source.lighting,
                            "endpoint": endpoint,
                            "startup": endpoint < 25,
                            "mse": float(mse),
                            "cosine": float(cosine),
                            "total": float(total),
                            "valid_tokens": int(valid),
                            "width": int(width),
                        }
                    )
    finally:
        set_full_visual_train_mode(student, original)
    if int(world) > 1:
        gathered: list[Any] = [None] * int(world)
        import torch.distributed as dist

        dist.all_gather_object(gathered, rows)
        rows = [row for part in gathered if part for row in part]

    def agg(selected: list[dict[str, Any]]) -> dict[str, Any]:
        valid = sum(r["valid_tokens"] for r in selected)
        if not valid:
            return {"windows": 0, "valid_tokens": 0}
        return {
            "windows": len(selected),
            "valid_tokens": valid,
            "feature_mse": sum(r["mse"] * r["valid_tokens"] for r in selected) / valid,
            "feature_cosine": sum(r["cosine"] * r["valid_tokens"] for r in selected) / valid,
            "feature_total": sum(r["total"] * r["valid_tokens"] for r in selected) / valid,
        }

    all_rows = agg(rows)
    return {
        "windows": len(rows),
        "valid_tokens": sum(r["valid_tokens"] for r in rows),
        "feature_mse": all_rows.get("feature_mse", 0.0),
        "feature_cosine": all_rows.get("feature_cosine", 0.0),
        "feature_total": all_rows.get("feature_total", 0.0),
        "startup": agg([r for r in rows if r["startup"]]),
        "eligible_ge_25": agg([r for r in rows if not r["startup"]]),
        "per_domain": {
            d: agg([r for r in rows if r["domain"] == d]) for d in ("libero", "robotwin2")
        },
        "per_lighting": {
            light: agg([r for r in rows if r["lighting"] == light])
            for light in sorted({r["lighting"] for r in rows})
        },
    }


def select_best(previous: float | None, current: float) -> tuple[float, bool]:
    """Return (best, improved); update only after comparison."""
    if previous is None or float(current) < float(previous):
        return float(current), True
    return float(previous), False


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state(rank: int) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(rank),
        "python": random.getstate(),
    }


def _restore_rng(state: Mapping[str, Any], rank: int) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], rank)
    random.setstate(state["python"])


def _disk_used(*roots: Path) -> int:
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def _enforce_disk(output: Path, temp: Path, cap_bytes: int, extra: int = 0) -> None:
    used = _disk_used(output, temp)
    if used + int(extra) > int(cap_bytes):
        raise AllBenchError(f"disk cap exceeded: used={used} extra={extra} cap={cap_bytes}")


def submit_cpu_samples(executor: Any, loader: Any, keys: Sequence[tuple[str, int]]) -> list[Any]:
    """Submit CPU-only full-sample construction, preserving sampler order."""
    return [executor.submit(loader, source_id, endpoint) for source_id, endpoint in keys]


def should_save_checkpoint(global_step: int) -> bool:
    """First full checkpoint at update 1, then every 250 updates."""
    step = int(global_step)
    return step == 1 or step % 250 == 0


def should_validate(global_step: int, *, epoch_end: bool, interval: int) -> bool:
    step = int(global_step)
    due = int(interval) > 0 and step > 0 and step % int(interval) == 0
    return bool(epoch_end or due)


def train_all_bench(
    config: Mapping[str, Any], args: Any, *, device: torch.device
) -> dict[str, Any]:
    """3-epoch distributed token training with complete resumable checkpoints."""
    import signal
    from concurrent.futures import ThreadPoolExecutor

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from owac.runtime.raw_full_visual_distillation import (
        _checkpoint_wrap_vision,
        set_full_visual_train_mode,
    )
    from owac.training.raw_distillation import token_distillation_loss
    from owac.training.raw_full_visual_batched import BatchedTemporalRequest, stack_teacher_targets

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.manual_seed(101)
    sources, _ = load_inventory(config["data"]["inventory"])
    working_sources = max(1, int(config["training"].get("working_cache_sources", 16)))
    dataset = AllBenchDataset(
        sources,
        targets=TargetStore(args.temp / "targets"),
        capacity=working_sources,
    )
    student = build_all_bench_student(config, device=str(device))
    if bool(config["training"].get("activation_checkpointing", True)):
        _checkpoint_wrap_vision(student)
    student.train(True)
    set_full_visual_train_mode(student, True)
    model = DDP(student, device_ids=[local], find_unused_parameters=False) if world > 1 else student
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    warmup = int(config["training"]["warmup_updates"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, float(step + 1) / max(1, warmup))
    )
    epochs = int(getattr(args, "epochs", 0) or config["training"]["epochs"])
    keys = dataset.anchor_keys("train")
    if not keys:
        raise AllBenchError("no train anchors in target store")
    start_epoch, start_step = 0, 0
    best_total: float | None = None
    paused = False
    paused_run = False

    def _pause(_signum: int, _frame: Any) -> None:
        nonlocal paused
        paused = True

    signal.signal(signal.SIGUSR1, _pause)
    if getattr(args, "resume", None):
        payload = torch.load(Path(args.resume) / "train.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch, start_step = int(payload["epoch"]), int(payload["step"])
        best_total = payload.get("best_total")
        if payload.get("rng_by_rank"):
            _restore_rng(payload["rng_by_rank"][rank], local)
        del payload
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "updates.jsonl"
    validation_path = output / "validation.jsonl"
    cap_bytes = int(float(getattr(args, "disk_cap_gib", 380.0)) * (1024**3))

    def save_state(epoch_value: int, step_value: int, best: float | None) -> None:
        local_rng = _rng_state(local)
        rng_by_rank: list[Any] = [None] * world
        if world > 1:
            dist.all_gather_object(rng_by_rank, local_rng)
        else:
            rng_by_rank[0] = local_rng
        if rank != 0:
            return
        _enforce_disk(output, Path(args.temp), cap_bytes, 6 * 1024**3)
        save = output / "checkpoints" / "latest"
        save.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch_value),
            "step": int(step_value),
            "best_total": best,
            "rng_by_rank": rng_by_rank,
            "global_step": int(global_step),
            "new_anchor_presentations": int(global_step) * 16,
            "seen_anchors": int(epoch_value) * len(keys),
        }
        atomic_save(save / "train.pt", payload)

    def validate_and_save(cursor_epoch: int, cursor_step: int) -> None:
        nonlocal best_total
        validation_student = model.module if world > 1 else model
        validation = validate_all_bench(
            validation_student, dataset, device=device, rank=rank, world=world
        )
        if rank == 0:
            with validation_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": int(global_step),
                            "epoch": int(cursor_epoch) + int(cursor_step > 0),
                            "global_step": int(global_step),
                            **validation,
                        }
                    )
                    + "\n"
                )
            total_value = float(validation["feature_total"])
            best_total, improved = select_best(best_total, total_value)
            if improved:
                best = output / "checkpoints" / "best"
                best.mkdir(parents=True, exist_ok=True)
                atomic_save(
                    best / "model.pt",
                    {
                        "model": model.state_dict(),
                        "feature_total": total_value,
                        "epoch": int(cursor_epoch) + int(cursor_step > 0),
                    },
                )
        if world > 1:
            value = torch.tensor(
                [float("inf") if best_total is None else float(best_total)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
            best_total = None if not torch.isfinite(value).item() else float(value.item())
        save_state(int(cursor_epoch), int(cursor_step), best_total)
        if world > 1:
            dist.barrier()

    if rank == 0 and not getattr(args, "resume", None):
        initial = output / "checkpoints" / "initial"
        initial.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, initial / "model.pt")
    global_step = (
        start_epoch
        * sum(len(_chunks_for_epoch(keys, dataset.sources, seed=101, epoch=e)) for e in range(1))
        + start_step
    )
    prefetch_workers = max(1, int(config["training"].get("prefetch_workers", 4)))
    sample_executor = ThreadPoolExecutor(max_workers=prefetch_workers)
    pending_samples: list[Any] | None = None
    for epoch in range(start_epoch, epochs):
        chunks = _chunks_for_epoch(keys, dataset.sources, seed=101, epoch=epoch)
        for step, chunk in enumerate(chunks):
            if epoch == start_epoch and step < start_step:
                continue
            step_started = time.monotonic()
            same = chunk[rank * 8 : (rank + 1) * 8]
            # A rank must always see exactly eight samples; tail chunks are padded.
            while len(same) < 8:
                same.append(chunk[-1])
            if pending_samples is None:
                pending_samples = submit_cpu_samples(sample_executor, dataset.sample, same)
            samples = [future.result() for future in pending_samples]
            if step + 1 < len(chunks):
                next_chunk = chunks[step + 1]
                next_same = next_chunk[rank * 8 : (rank + 1) * 8]
                while len(next_same) < 8:
                    next_same.append(next_chunk[-1])
                pending_samples = submit_cpu_samples(sample_executor, dataset.sample, next_same)
            else:
                pending_samples = None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model(BatchedTemporalRequest(tuple(samples)))
                teacher, mask = stack_teacher_targets(samples)
                combined = result.token_mask & mask.to(result.token_mask.device)
                loss = token_distillation_loss(result.projected_tokens, teacher, combined)
                total = loss.total
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                params, float(config["training"].get("gradient_clip", 1.0))
            )
            lr_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            pause_tensor = torch.tensor(int(paused), device=device, dtype=torch.int32)
            if world > 1:
                dist.all_reduce(pause_tensor, op=dist.ReduceOp.MAX)
            paused_any = bool(pause_tensor.item())
            if should_save_checkpoint(global_step) or paused_any:
                save_state(epoch, step + 1, best_total)
            if paused_any:
                if rank == 0:
                    (output / "run-status.json").write_text(
                        json.dumps(
                            {
                                "status": "paused",
                                "global_step": global_step,
                                "epoch": epoch + 1,
                                "step": step + 1,
                            },
                            indent=2,
                        )
                    )
                break
            if world > 1:
                for value in (loss.mse, loss.cosine, loss.total):
                    dist.all_reduce(value.detach(), op=dist.ReduceOp.SUM)
                loss_values = (loss.mse / world, loss.cosine / world, loss.total / world)
            else:
                loss_values = (loss.mse, loss.cosine, loss.total)
            if rank == 0 and (global_step % 10 == 0 or step == len(chunks) - 1):
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "loss_components": {
                                    "mse": float(loss_values[0]),
                                    "cosine": float(loss_values[1]),
                                    "total": float(loss_values[2]),
                                },
                                "lr_used": lr_used,
                                "new_global_updates": global_step,
                                "new_anchor_presentations": global_step * 16,
                                "effective_batch": 16,
                                "epoch": epoch + 1,
                                "host_wall_phases": {
                                    "total_seconds": time.monotonic() - step_started
                                },
                                "seen_anchors": (epoch * len(keys))
                                + min((step + 1) * 16, len(keys)),
                            }
                        )
                        + "\n"
                    )
            epoch_end = step == len(chunks) - 1
            interval = int(config["training"].get("validation_interval_updates", 500))
            if should_validate(global_step, epoch_end=epoch_end, interval=interval):
                if epoch_end:
                    validate_and_save(epoch + 1, 0)
                else:
                    validate_and_save(epoch, step + 1)
        if paused_any:
            paused_run = True
            break
    if rank == 0:
        _enforce_disk(output, Path(args.temp), cap_bytes, 6 * 1024**3)
        final = output / "checkpoints" / "final"
        final.mkdir(parents=True, exist_ok=True)
        latest_file = output / "checkpoints" / "latest" / "train.pt"
        if latest_file.is_file():
            (final / "train.pt").unlink(missing_ok=True)
            os.link(latest_file, final / "train.pt")
        else:
            torch.save({"model": model.state_dict()}, final / "train.pt")
        (output / "run-status.json").write_text(
            json.dumps(
                {
                    "status": "paused" if paused_run else "completed",
                    "epochs": epochs,
                    "global_updates": global_step,
                    "best_feature_total": best_total,
                    "unique_train_anchors": len(keys),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    sample_executor.shutdown(wait=True)
    if world > 1:
        dist.destroy_process_group()
    return {"epochs": epochs, "global_updates": global_step, "unique_train_anchors": len(keys)}
