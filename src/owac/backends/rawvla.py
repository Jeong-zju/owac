"""Audited checkpoint loading for the pinned RawVLA StarVLA implementation.

Run with the external RawVLA environment. Heavy dependencies stay command-local.
LIBERO uses the upstream OpenPI tensor conversion functions without importing its
training stack. RoboTwin loads the author's original LeRobot tensors directly.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audited(model, state: dict) -> dict:
    """Reject unmatched weights; allow only upstream's inactive adaRMS scales."""
    state = model._normalize_state_dict(state)
    current = model.state_dict()
    embed = "vlm.model.model.language_model.embed_tokens.weight"
    head = "vlm.model.lm_head.weight"
    if head in current and head not in state and embed in state:
        state[head] = state[embed]
    unexpected = sorted(state.keys() - current.keys())
    mismatches = [
        key for key in state.keys() & current.keys() if state[key].shape != current[key].shape
    ]
    missing = sorted(current.keys() - state.keys())
    inactive = [key for key in missing if model._is_expected_openpi_missing_key(key)]
    # OpenPIActionHead uses gemma_expert.model and action_out_proj; its language
    # vocabulary head is never called. OpenPI/LeRobot do not supply this tensor.
    unused_head = "action_head.gemma_expert.lm_head.weight"
    if unused_head in missing:
        inactive.append(unused_head)
        model.action_head.gemma_expert.lm_head.weight.data.zero_()
    active_missing = sorted(set(missing) - set(inactive))
    audit = {
        "loaded_keys": len(state),
        "model_keys": len(current),
        "inactive_keys": inactive,
        "active_missing": active_missing,
        "unexpected": unexpected,
        "shape_mismatch": mismatches,
    }
    if active_missing or unexpected or mismatches:
        raise ValueError(f"Checkpoint does not match pinned StarVLA: {audit}")
    result = model.load_state_dict(state, strict=False)
    if sorted(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("Checkpoint audit and load results differ")
    return audit


def new_model(suite: str, tokenizer: Path):
    from omegaconf import OmegaConf
    from starVLA.model.framework.VLM4A.PI05 import PI05

    config = OmegaConf.create(
        {
            "framework": {
                "name": "PI05",
                "precision": "bfloat16",
                "action_horizon": 10 if suite == "libero" else 50,
                "discrete_state_input": suite != "libero",
                "tokenizer": {"model_path": str(tokenizer), "pad_state_value": 0.0},
            },
            "trainer": {"pretrained_checkpoint": None},
        }
    )
    return PI05(config)


def restore_libero(source: Path, converter: Path) -> dict:
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    import torch
    from flax import traverse_util

    tree = ast.parse(converter.read_text())
    names = {"slice_paligemma_state_dict", "slice_gemma_state_dict"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(functions) != 2:
        raise ValueError("Expected the two pinned OpenPI conversion functions")
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(converter), "exec"), namespace)
    with ocp.PyTreeCheckpointer() as checkpointer:
        item = {"params": checkpointer.metadata(source / "params")["params"]}
        params = checkpointer.restore(
            source / "params",
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32), item
                ),
            ),
        )["params"]
    flat = traverse_util.flatten_dict(params)
    if all(key[-1] == "value" for key in flat):
        flat = {key[:-1]: value for key, value in flat.items()}
    params = traverse_util.unflatten_dict(flat)
    config = SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27),
        text_config=SimpleNamespace(
            hidden_size=2048, num_hidden_layers=18, num_attention_heads=8, head_dim=256
        ),
    )
    pali, expert = namespace["slice_paligemma_state_dict"](
        traverse_util.flatten_dict(params["PaliGemma"], sep="/"), config
    )
    gemma = namespace["slice_gemma_state_dict"](
        expert,
        SimpleNamespace(width=1024, depth=18, num_heads=8, head_dim=256),
        num_expert=1,
        checkpoint_dir="pi05_libero",
        pi05=True,
    )
    projections = {}
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        projections[f"{name}.weight"] = torch.from_numpy(params[name]["kernel"]).T
        projections[f"{name}.bias"] = torch.from_numpy(params[name]["bias"])
    return {**pali, **gemma, **projections}


def prepare_libero(source: Path, converter: Path, tokenizer: Path, output: Path) -> None:
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    if output.exists():
        raise FileExistsError(output)
    state = restore_libero(source.resolve(), converter)
    model = new_model("libero", tokenizer)
    audit = load_audited(model, state)
    output.mkdir(parents=True)
    destination = output / "bfloat16/model.safetensors"
    destination.parent.mkdir()
    # Clone tied aliases so StarVLA's strict loader can read every named tensor.
    save_file(
        {key: tensor.contiguous().clone() for key, tensor in model.state_dict().items()},
        str(destination),
    )
    OmegaConf.save(model.config, output / "config.yaml")
    stats_path = source / "assets/physical-intelligence/libero/norm_stats.json"
    stats = json.loads(stats_path.read_text())["norm_stats"]
    (output / "dataset_statistics.json").write_text(
        json.dumps(
            {"openpi_pi05_libero_franka": {"state": stats["state"], "action": stats["actions"]}},
            indent=2,
        )
        + "\n"
    )
    audit.update(
        converter_sha256=sha256_file(converter),
        original_stats_sha256=sha256_file(stats_path),
        model_sha256=sha256_file(destination),
        tokenizer_sha256=sha256_file(tokenizer),
        extra_training_steps=0,
    )
    (output / "load-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


def serve_libero(checkpoint: Path, tokenizer: Path, port: int) -> None:
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    from server_starvla_openpi import StarVLAOpenPIPolicy

    policy = StarVLAOpenPIPolicy("PI05", checkpoint, tokenizer, "bfloat16", "cuda", "openpi")
    WebsocketPolicyServer(
        policy=policy, host="127.0.0.1", port=port, idle_timeout=-1, metadata=policy.metadata
    ).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-libero")
    convert.add_argument("source", type=Path)
    convert.add_argument("converter", type=Path)
    convert.add_argument("tokenizer", type=Path)
    convert.add_argument("output", type=Path)
    serve = commands.add_parser("serve-libero")
    serve.add_argument("--checkpoint", required=True, type=Path)
    serve.add_argument("--tokenizer", required=True, type=Path)
    serve.add_argument("--port", default=18000, type=int)
    args = parser.parse_args()
    if args.command == "convert-libero":
        prepare_libero(args.source, args.converter, args.tokenizer, args.output)
    else:
        serve_libero(args.checkpoint, args.tokenizer, args.port)


if __name__ == "__main__":
    main()
