"""Bounded real-data r4 acceptance; no policy build and no optimizer construction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from owac.backends.n02_v12_tokens import N02V12TokenSource
from owac.evaluation.n02_full import v12 as evaluation
from owac.frameworks.n02_full.model_v12 import build_n02_v12_model
from owac.frameworks.n02_full.sequence_v12 import EXECUTION_REVISION, window_inputs
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, build_source, resolve_plan
from owac.training.n02_v12_profile import parameter_digest

parser = argparse.ArgumentParser()
parser.add_argument("--platform", choices=("robotwin", "libero"), required=True)
args = parser.parse_args()
config_path = f"configs/encoders/n02_v12/raw_n02_{args.platform}.json"
plan = resolve_plan(build_parser().parse_args(["--config", config_path]))
torch.set_num_threads(4)
torch.manual_seed(plan.seed)
np.random.seed(plan.seed)
trainer = N02V12Trainer(plan, LogTail())
trainer.device = torch.device("cuda")
trainer.representation = plan.config["representation"]
trainer.sources = {
    part: build_source(plan.config, part, history=plan.history, batch_size=1)
    for part in ("train", "dev", "calibration")
}
trainer.model = build_n02_v12_model(plan.config["model"]).to(trainer.device).eval()
parameters = list(trainer.model.parameters())
before = parameter_digest(parameters)
checks = trainer.correctness_checks()
source = N02V12TokenSource(trainer.model, camera_ids=plan.config["model"]["camera_ids"], history=8)
records = []
with torch.no_grad():
    for index in evaluation.anchor_indices(trainer.sources["train"], 3):
        batch = trainer.window(trainer.sources["train"], index, index + 1)
        tokens = source(batch).values
        reference = evaluation.replay_window(trainer.model, batch, device="cuda", check_label=0)
        delta = float((tokens - reference["rows"][-1].tokens).abs().max())
        assert delta == 0
        queries = trainer.model.inference.build_queries(window_inputs(batch)[-1], batch=1)
        valid = queries.masks["valid"] & ~queries.masks["bad"] & ~queries.masks["missing"]
        scorable = valid & ~queries.masks["saturated"]
        expected = int(scorable.sum())
        scored = reference["rows"][-1].diagnostics["scored_scalars_per_candidate"][0]
        assert all(count == expected for count in scored)
        records.append(
            {
                "anchor_index": index,
                "train_eval_token_max_abs_delta": delta,
                "total_scalar_queries": queries.target.numel(),
                "data_valid_scalars": int(valid.sum()),
                "saturated_excluded_scalars": int((valid & queries.masks["saturated"]).sum()),
                "scorable_scalars": expected,
                "actual_scored_per_candidate": scored,
                "frame_indices": batch.frame_index.tolist(),
                "anchors": list(batch.anchors),
            }
        )
prior = trainer.prior_wiring_check()
train_calibration = trainer.calibration_pass()
settings = evaluation.resolve_settings(
    evaluation.build_parser().parse_args(
        [
            "--config",
            config_path,
            "--calibration-samples",
            "4",
            "--dev-samples",
            "4",
            "--action-check",
            "skip",
            "--no-figures",
            "--no-interventions",
            "--device",
            "cuda",
        ]
    )
)
with torch.no_grad():
    eval_calibration = evaluation.calibration_pass(trainer.model, settings, device="cuda")
eval_threshold = float(eval_calibration["_calibration"].calibrator.threshold())
delta_threshold = abs(float(train_calibration["threshold"]) - eval_threshold)
assert delta_threshold == 0
assert before == parameter_digest(parameters)
output = {
    "execution_revision": EXECUTION_REVISION,
    "platform": args.platform,
    "initialisation": "seed1103 fresh full frontend; no backend constructed",
    "optimizer_constructed": False,
    "optimizer_updates": 0,
    "frontend_weights_unchanged": True,
    "frontend_sha256": before,
    "checks": checks,
    "coverage_and_tokens": records,
    "prior": prior,
    "training_calibration": train_calibration,
    "eval_calibration_threshold": eval_threshold,
    "threshold_max_abs_delta": delta_threshold,
    "claim_boundary": (
        "fixed fresh weights, four calibration windows; "
        "consistency only, no calibration validity or capability claim"
    ),
}
path = Path(f"outputs/n02-v12/P2/fresh-acceptance-r4-{args.platform}.json")
path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n")
print(
    json.dumps({"output": str(path), "threshold_delta": delta_threshold, "optimizer_updates": 0}),
    flush=True,
)
