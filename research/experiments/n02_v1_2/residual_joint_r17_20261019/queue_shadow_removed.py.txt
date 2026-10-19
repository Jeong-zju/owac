"""Foreground supervisor dependency gate; no training before data acceptance."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[3] / "outputs/n02-v12/residual-joint-r17"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "train"])
    p.add_argument("--seed", type=int, default=17019)
    a = p.parse_args()
    start = time.time()
    while True:
        if time.time() - start > 3600:
            raise TimeoutError("dependency exceeded fixed1hour wait")
        if a.command == "prepare":
            files = list((OUT / "receipts").glob("*.json"))
            if len(files) == 20:
                break
        else:
            errors = list(OUT.glob("prepare-*-error.json"))
            if errors:
                raise RuntimeError("preparation failed; no training")
            f = OUT / "prepare/status.json"
            if f.exists() and json.loads(f.read_text()).get("state") == "completed":
                break
        time.sleep(5)
    os.execv(
        sys.executable, [sys.executable, str(HERE / "run.py"), a.command, "--seed", str(a.seed)]
    )
