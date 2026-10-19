"""Build the synchronized RAW -> action timeline for a real rollout.

The required deliverable is one time axis carrying the RAW frame, its frozen-ISP RGB, the
support state the model decided from, the fusion weights it used, and the actions that
were executed.  Everything here comes from a recording of a real rollout rather than a
replay, because the support decision and the fusion weights are computed on the way to the
action and a replay would have to assume a checkpoint that may no longer be the one that
produced the action.

The ISP is applied here rather than in the client: the client's PYTHONPATH resolves a
different ``owac`` that has no ``owac.raw``, so the same frozen function is called from the
side where the import is guaranteed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.raw.n01_paired_views import paired_views


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def group_by_identity(records: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        key = (record.get("task_suite"), record.get("task_id"), record.get("episode_idx"))
        grouped.setdefault(key, []).append(record)
    for sequence in grouped.values():
        sequence.sort(key=lambda item: item.get("query", 0))
    return grouped


def _isp_rgb(raw_path: Path) -> np.ndarray | None:
    """Apply the same frozen ISP the service used, to the recorded RAW frame."""

    from PIL import Image

    if not raw_path.is_file():
        return None
    array = np.asarray(Image.open(raw_path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.as_tensor(array)[None, None]
    view = getattr(paired_views(tensor, platform="libero"), "rgb", None)
    if view is None:
        return None
    result = view.detach().cpu().numpy()[0, 0]
    if result.ndim == 3 and result.shape[0] in (1, 3):
        result = np.transpose(result, (1, 2, 0))
    return np.clip(result, 0.0, 1.0)


def _usable_fraction(diagnostics: Any) -> float | None:
    """Share of history *cells* the support mechanism kept usable for this query.

    Not a share of sources: ``source_usable`` is a per-cell mask, so this counts usable
    cells over all history slots and cells.  The two differ whenever a region is only
    partly usable, and naming it after sources invited reading it as a source count.
    """

    if not isinstance(diagnostics, dict):
        return None
    support = diagnostics.get("support") or {}
    # The service reports counts plus a bounded sample; the counts are exact.
    cells = support.get("usable_cells")
    total = support.get("total_cells")
    if cells is not None and total:
        return float(cells) / float(total)
    usable = support.get("source_usable_sample") or support.get("source_usable")
    if usable is None:
        return None
    flat: list[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif value is not None:
            flat.append(value)

    walk(usable)
    if not flat:
        return None
    return sum(1 for item in flat if item) / len(flat)


def _sample_indices(count: int, wanted: int) -> list[int]:
    if count <= wanted:
        return list(range(count))
    step = max(count // wanted, 1)
    return list(range(0, count, step))[:wanted]


def build_figure(
    records_path: Path,
    identity: tuple,
    output: Path,
    *,
    samples: int = 6,
) -> dict[str, Any]:
    """Render the synchronized timeline figure for one identity."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = group_by_identity(load_records(records_path))
    if identity not in grouped:
        raise SystemExit(f"identity {identity} is not in {records_path}")
    sequence = grouped[identity]

    raw_dir = records_path.with_name(records_path.name.replace(".jsonl", "-raw"))

    picked = _sample_indices(len(sequence), samples)
    figure, axes = plt.subplots(
        4,
        max(len(picked), 1),
        figsize=(2.0 * max(len(picked), 1), 9.2),
        gridspec_kw={"height_ratios": [1, 1, 0.9, 1.5]},
    )
    if len(picked) == 1:
        axes = axes.reshape(4, 1)

    for column, index in enumerate(picked):
        record = sequence[index]
        raw_name = record.get("raw_path")
        raw_image = None
        isp_image = None
        if raw_name:
            from PIL import Image

            raw_file = raw_dir / raw_name
            if raw_file.is_file():
                raw_image = np.asarray(Image.open(raw_file).convert("RGB"), dtype=np.uint8)
                isp_image = _isp_rgb(raw_file)

        axes[0, column].imshow(raw_image)
        axes[0, column].set_title(f"q{record.get('query')}", fontsize=8)
        axes[0, column].axis("off")
        if column == 0:
            axes[0, column].set_ylabel("RAW", fontsize=9)

        axes[1, column].imshow(isp_image)
        axes[1, column].axis("off")
        if column == 0:
            axes[1, column].set_ylabel("frozen ISP", fontsize=9)

        fraction = _usable_fraction(record.get("diagnostics"))
        axes[2, column].bar(
            [0], [0.0 if fraction is None else fraction], color="#3b6ea5", width=0.6
        )
        axes[2, column].set_ylim(0, 1)
        axes[2, column].set_xticks([])
        text = "n/a" if fraction is None else f"{fraction:.2f}"
        axes[2, column].text(0, 0.05, text, ha="center", fontsize=8)
        if column == 0:
            axes[2, column].set_ylabel("usable cells", fontsize=9)

    # Fusion weights of the newest sampled query: which history sources carried weight.
    axes[3, 0].set_ylabel("fusion weight", fontsize=9)
    last = sequence[picked[-1]].get("diagnostics") or {}
    weights = (last.get("fusion") or {}).get("weights") or []
    if weights:
        values = np.asarray(weights, dtype=np.float32).reshape(-1)
        axes[3, 0].bar(np.arange(values.size), values, width=1.0, color="#8a5a3b")
        # The service ships a bounded strided sample of the weight tensor, not one entry
        # per source, so the axis is a sample index and the count is over sampled values.
        # Calling it a slot-per-source axis with an "N/M sources" title overstated what the
        # payload actually contains.
        axes[3, 0].set_xlabel("sampled weight index (strided payload)", fontsize=8)
        axes[3, 0].tick_params(labelsize=7)
        nonzero = int((np.abs(values) > 1e-6).sum())
        axes[3, 0].set_title(f"{nonzero}/{values.size} sampled weights non-zero", fontsize=8)
    else:
        axes[3, 0].text(0.5, 0.5, "no fusion weights recorded", ha="center", fontsize=8)
        axes[3, 0].set_xticks([])

    # Executed actions across the whole episode, on a real time axis.
    # A missing action stays missing (NaN) rather than becoming a plotted zero: a
    # fabricated zero is an unmeasured value presented as a measurement.
    actions = np.asarray(
        [
            (
                np.asarray(record["actions"])[0]
                if record.get("actions")
                else np.full(7, np.nan, dtype=np.float32)
            )
            for record in sequence
        ],
        dtype=np.float32,
    )
    for dim in range(actions.shape[1]):
        axes[3, 1].plot(np.arange(actions.shape[0]), actions[:, dim], linewidth=1.0)
    axes[3, 1].set_xlabel("policy query", fontsize=8)
    axes[3, 1].set_ylabel("executed action", fontsize=8)
    axes[3, 1].tick_params(labelsize=7)
    axes[3, 1].set_title("first step of each returned chunk", fontsize=8)
    for column in range(2, max(len(picked), 1)):
        axes[3, column].axis("off")

    suite, task, episode = identity
    usable_values = [
        value
        for value in (_usable_fraction(record.get("diagnostics")) for record in sequence)
        if value is not None
    ]
    figure.suptitle(
        f"N01 v1.2 P3 synchronized timeline - {suite} task {task} episode {episode}\n"
        f"{len(sequence)} policy queries, recorded from a real rollout; "
        f"usable-cell share range {min(usable_values):.2f}-{max(usable_values):.2f}"
        if usable_values
        else f"N01 v1.2 P3 synchronized timeline - {suite} task {task} episode {episode}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)

    return {
        "identity": list(identity),
        "queries": len(sequence),
        "sampled": [sequence[index].get("query") for index in picked],
        "raw_frames_available": sum(
            1 for r in sequence if (raw_dir / (r.get("raw_path") or "")).is_file()
        ),
        "fusion_sampled_weights_non_zero": int(
            (np.abs(np.asarray(weights, dtype=np.float32)).reshape(-1) > 1e-6).sum()
        )
        if weights
        else 0,
        "fusion_sampled_weights_total": (
            int(np.asarray(weights, dtype=np.float32).size) if weights else 0
        ),
        "fusion_note": (
            "the weights are a bounded strided sample of the fusion tensor, so these count "
            "sampled values, not history sources"
        ),
        "usable_share_range": [min(usable_values), max(usable_values)] if usable_values else None,
        "figure": str(output),
    }


def video_outcome(identity: tuple, episode_ledger: Path | None = None) -> str:
    """Query recordings contain no terminal success signal; read the episode ledger."""
    if episode_ledger is None:
        return "  (outcome unavailable: no episode ledger)"
    from owac.runtime.n01_v12_libero_dev import read_episode_ledger

    record = read_episode_ledger(episode_ledger).get(identity)
    if record is None:
        return "  (outcome unavailable: episode not completed)"
    return "  (recorded outcome: SUCCESS)" if record["success"] else "  (recorded outcome: FAILURE)"


def build_video(
    records_path: Path,
    identity: tuple,
    output: Path,
    *,
    fps: int = 6,
    episode_ledger: Path | None = None,
) -> dict[str, Any]:
    """Render one real rollout as a video: RAW, its frozen-ISP RGB, and the action trace.

    The task asks for a closed-loop video beside the case figures.  Because both the RAW
    frames and the executed actions were recorded during a real rollout, the video is
    assembled from those recordings rather than re-simulated, so the action trace is the
    control that was actually sent.

    Terminal success is read separately from the completed episode ledger. Query
    records without that signal cannot establish a success or failure count.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    grouped = group_by_identity(load_records(records_path))
    if identity not in grouped:
        raise SystemExit(f"identity {identity} is not in {records_path}")
    sequence = grouped[identity]
    raw_dir = records_path.with_name(records_path.name.replace(".jsonl", "-raw"))
    outcome = video_outcome(identity, episode_ledger)

    # A missing action stays missing (NaN) rather than becoming a plotted zero: a
    # fabricated zero is an unmeasured value presented as a measurement.
    actions = np.asarray(
        [
            (
                np.asarray(record["actions"])[0]
                if record.get("actions")
                else np.full(7, np.nan, dtype=np.float32)
            )
            for record in sequence
        ],
        dtype=np.float32,
    )
    frames_dir = output.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, record in enumerate(sequence):
        raw_name = record.get("raw_path")
        if not raw_name:
            continue
        raw_file = raw_dir / raw_name
        if not raw_file.is_file():
            continue
        raw_image = np.asarray(Image.open(raw_file).convert("RGB"), dtype=np.uint8)
        isp_image = _isp_rgb(raw_file)

        figure = plt.figure(figsize=(7.2, 3.6), dpi=110)
        grid = figure.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.35, wspace=0.05)
        first = figure.add_subplot(grid[0, 0])
        second = figure.add_subplot(grid[0, 1])
        first.imshow(raw_image)
        first.set_title("RAW", fontsize=9)
        second.imshow(isp_image)
        second.set_title("frozen ISP-RGB", fontsize=9)
        for axis in (first, second):
            axis.axis("off")

        trace = figure.add_subplot(grid[1, :])
        for dim in range(actions.shape[1]):
            trace.plot(np.arange(actions.shape[0]), actions[:, dim], linewidth=0.9)
        trace.axvline(index, color="black", linewidth=1.2)
        trace.set_xlim(0, max(actions.shape[0] - 1, 1))
        trace.set_xlabel("policy query", fontsize=8)
        trace.set_ylabel("executed action", fontsize=8)
        trace.tick_params(labelsize=7)
        trace.set_title("executed action, cursor = now", fontsize=8)

        usable = _usable_fraction(record.get("diagnostics"))
        suite, task, episode = identity
        figure.suptitle(
            f"{suite} task {task} episode {episode} - query {index + 1}/{len(sequence)}"
            + ("" if usable is None else f" - usable cells {usable:.2f}")
            + outcome,
            fontsize=9,
        )
        frame_path = frames_dir / f"frame{index:04d}.png"
        figure.savefig(frame_path)
        plt.close(figure)
        written += 1

    if not written:
        raise SystemExit("no frames were written; nothing to encode")
    import subprocess

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "identity": list(identity),
        "frames": written,
        "fps": fps,
        "video": str(output),
        "encoded": completed.returncode == 0,
        "ffmpeg_tail": completed.stderr.strip().splitlines()[-1] if completed.stderr else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the synchronized rollout timeline")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--video",
        default=None,
        help="also render an MP4 of this identity's real rollout (needs ffmpeg)",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--episode-ledger", type=Path, default=None)
    args = parser.parse_args(argv)

    records_path = Path(args.records)
    grouped = group_by_identity(load_records(records_path))
    if not grouped:
        raise SystemExit(f"no usable records in {records_path}")
    if args.suite is None:
        identity = max(grouped, key=lambda key: len(grouped[key]))
    else:
        identity = (args.suite, args.task, args.episode)
    receipt = build_figure(records_path, identity, Path(args.output), samples=args.samples)
    if args.video:
        video_receipt = build_video(
            records_path,
            identity,
            Path(args.video),
            fps=args.fps,
            episode_ledger=args.episode_ledger,
        )
        receipt["video"] = video_receipt
        print(json.dumps(video_receipt, ensure_ascii=False))
    if args.report:
        Path(args.report).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
