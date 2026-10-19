"""Final P2 integration probes: real preview, shared calibration and real policy sampling."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from owac.evaluation.n02_full import v12 as ev
from owac.frameworks.n02_full.checks import sequence_statistic
from owac.frameworks.n02_full.data_v12 import apply_declared_isp
from owac.frameworks.n02_full.sequence_v12 import replay_inputs, window_inputs
from owac.training.n02_v12 import LogTail, N02V12Trainer, build_parser, resolve_plan

parser = argparse.ArgumentParser()
parser.add_argument("platform", choices=("robotwin", "libero"))
args = parser.parse_args()
root = Path.cwd()
out = root / "outputs/n02-v12/P2/independent-audit-20261016" / args.platform
settings = ev.resolve_settings(
    ev.build_parser().parse_args(
        [
            "--config",
            f"configs/encoders/n02_v12/raw_n02_{args.platform}.json",
            "--output",
            str(out / "final-probe"),
            "--no-publish-status",
        ]
    )
)
receipt = {}
if args.platform == "robotwin":
    torch.set_num_threads(4)
    model = ev.build_model(settings, device="cuda")
    receipt["checkpoint"] = ev.load_checkpoint_into(model, out / "checkpoint_0000002.pt")
    source = ev.build_source(settings, "calibration")
    index = source.anchor_count // 2
    trainer = N02V12Trainer.__new__(N02V12Trainer)
    trainer.model, trainer.device, trainer.run_dir = model, torch.device("cuda"), out
    trainer.plan = SimpleNamespace(previews=1, calibration_windows=2, dev_windows=2)
    trainer.sources = {"calibration": source, "dev": ev.build_source(settings, "dev")}
    trainer.current_step = 2
    trainer.representation = {"kind": "raw3"}
    with torch.no_grad():
        batch = source.window(index, index + 1)
        replay = replay_inputs(model, window_inputs(batch))
        final = replay["rows"][-1]
        rule = ev.build_check_rule(settings)
        a = trainer._residual_views_of_window(source, index, rule)
        b = ev.locked_residual_groups(
            model.inference, replay["inputs"][-1], replay["pre_states"][-1], final.q
        )["groups"][0]
        receipt["same_sample_calibration_statistic_abs_delta"] = abs(
            sequence_statistic(a, rule=rule) - sequence_statistic(b, rule=rule)
        )
        receipt["previews"] = trainer.write_previews(batch, final)
        training_calibration = trainer.calibration_pass()
        settings.calibration_samples = 2
        evaluation_calibration = ev.calibration_pass(model, settings, device="cuda")
        receipt["train_calibration_threshold"] = training_calibration["threshold"]
        receipt["eval_calibration_threshold"] = evaluation_calibration["calibration"]["threshold"]
        receipt["threshold_abs_delta"] = abs(
            receipt["train_calibration_threshold"] - receipt["eval_calibration_threshold"]
        )
        receipt["viewing_rule_hash_equal"] = (
            training_calibration["viewing_rule_hash"] == evaluation_calibration["rule_hash"]
        )

else:
    # Original LIBERO checkpoint remains frozen, front end is a fresh complete Base instance.
    plan = resolve_plan(
        build_parser().parse_args(
            [
                "--config",
                "configs/encoders/n02_v12/raw_n02_libero.json",
                "--max-updates",
                "1",
                "--output",
                str(out / "sampling-probe"),
            ]
        )
    )
    trainer = N02V12Trainer(plan, LogTail())
    trainer.build()
    source = trainer.sources["dev"]
    index = source.anchor_count // 2
    batch = trainer.window(source, index, index + 1)
    prompts = ev.official_prompts(settings, [source.anchor(index).as_dict()])
    proprio = trainer._anchor_proprio(batch)
    lang, mask, state = trainer.language(prompts, proprio)
    ev_lang, ev_mask, lang_info = ev._language_batch(trainer.bridge, prompts, proprio, settings)
    receipt["official_prompt"] = prompts
    receipt["train_eval_language_tokens_equal"] = bool(torch.equal(lang.cpu(), ev_lang))
    receipt["train_eval_language_masks_equal"] = bool(torch.equal(mask.cpu(), ev_mask))
    prediction = trainer.bridge.predict_action(
        observation=batch, lang_tokens=lang, lang_masks=mask, state=state
    )
    control = torch.as_tensor(prediction["control_actions"])
    receipt["normalized_sample_shape"] = list(control.shape)
    receipt["sample_finite"] = bool(torch.isfinite(control).all())
    receipt["previews"] = trainer.write_previews(batch, trainer.token_source.last_output)
    receipt["frozen_parameters_with_grad"] = [
        name
        for name, parameter in trainer.bridge.base.named_parameters()
        if parameter.grad is not None
    ]
    receipt["closed_loop_completed"] = 0
    receipt["note"] = "normalized policy sample only; no simulator action and no optimizer update"
# Check RGB evaluation source matches the training transform on this same official sample.
rgb_settings = ev.resolve_settings(
    ev.build_parser().parse_args(
        [
            "--config",
            f"configs/encoders/n02_v12/rgb_n02_{args.platform}.json",
        ]
    )
)
rgb_source = ev.build_source(rgb_settings, "dev")
rgb_index = rgb_source.anchor_count // 2
raw_batch = rgb_source.raw_window(rgb_index, rgb_index + 1)
rgb_batch = rgb_source.window(rgb_index, rgb_index + 1)
expected = apply_declared_isp(raw_batch.observation, rgb_settings.representation)
receipt["rgb_train_eval_input_max_abs_delta"] = max(
    float(
        (
            torch.as_tensor(expected.frames[c].values)
            - torch.as_tensor(rgb_batch.observation.frames[c].values)
        )
        .abs()
        .max()
    )
    for c in expected.frames
)
(out / "final_probe_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
