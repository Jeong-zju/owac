"""Independent all-development-frame check of a frozen RAW representation."""

import argparse
from pathlib import Path

import torch

from owac.frameworks.n02_full.model_v12 import build_n02_v12_model
from owac.frameworks.n02_full.representation_v12 import RawPatchHead
from owac.runtime.rawvla_full_eval import write_json
from owac.training.n02_v12_representation import evaluate, frame_pool


def main(args):
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_n02_v12_model(config["model"]).cuda()
    source = torch.load(
        config["initialisation"]["checkpoint"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(source["model"], strict=True)
    model.inference.encoder.load_state_dict(payload["encoder"], strict=True)
    head = RawPatchHead().cuda()
    head.load_state_dict(payload["raw_head"], strict=True)
    pool, frames, identity = frame_pool(config, "dev", per_episode=4)
    assert identity == payload["data_identity"]["dev"]
    report = evaluate(
        model, head, pool, tuple(config["model"]["camera_ids"]), "cuda", indices=range(len(pool))
    )
    write_json(Path(args.output), {"optimizer_updates": 0, "frames": frames, "metrics": report})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
