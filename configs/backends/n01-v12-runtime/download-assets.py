"""Restore public training assets; credentials come only from the HF token store.

Run using the server's download environment. No research training is started.
The GCS object list comes from the archived original deployment, including generations.
"""

import argparse
import concurrent.futures
import json
import os
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path("/workspace/owac/rawvla")


def dataset():
    snapshot_download(
        "ToferFish/RawVLA-Bench",
        repo_type="dataset",
        revision="8586c594fff9ce38580e29745f074e82e96e74f7",
        local_dir=ROOT / "source/benchmark_data/RawVLA-Bench",
        max_workers=16,
    )


def robotwin():
    snapshot_download(
        "SidneyXie/pi05_robotwin",
        revision="e49e2ab6c11f07511573b67261bd129e88d0a416",
        local_dir=ROOT / "checkpoints/pi05_robotwin",
        max_workers=4,
    )


def libero():
    archive = ROOT / "records/baseline-archive/remote/records/pi05-libero-gcs.json"
    items = json.loads(archive.read_text())["items"]

    def fetch(item):
        relative = item["name"].removeprefix("checkpoints/pi05_libero/")
        target = ROOT / "checkpoints/pi05_libero_jax" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == int(item["size"]):
            return
        temporary = target.with_name(target.name + ".download")
        # Pinned generation preserves the actual source, without adding hash gates.
        urllib.request.urlretrieve(item["mediaLink"], temporary)
        temporary.replace(target)
        print("downloaded", relative, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, items))
    tokenizer = ROOT / "source/openpi_converted_protocol/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    if not tokenizer.exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", tokenizer
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("dataset", "robotwin", "libero"))
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    globals()[args.asset]()
    print(args.asset, "download complete", flush=True)
