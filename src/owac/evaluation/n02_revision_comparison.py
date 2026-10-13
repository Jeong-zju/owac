"""Conditional audit of N02's local revision scheduler with reliable measurements.

This isolates the shipped semantic_map and one-hop dependency propagation with
an EXACT shared state-recomposition operator. It is not an end-to-end learned
N02 acceptance. Nonoverlapping 64x64 patches are observation supports, not pixels
onto which a global score is copied. All four methods receive the same scored
RAW, absolute-fit veto, priors and recorded dependency graph.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from owac.baselines.n02_measurement_adapters import AdaptedMeasurementScorer
from owac.core.encoder_contract import state_tensor_bytes
from owac.evaluation.n02_measurement_isolation import file_digest, tensor_digest, write_json
from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.hypotheses import NEIGHBOURHOOD, neighbour_stack
from owac.frameworks.n02.revision import RevisionController
from owac.frameworks.n02.synthetic import SyntheticGenerator, SyntheticOptions

METHODS = ("standard_filter", "global_update", "n02_local", "event_cached_filter")


def protocol(smoke=False):
    return {
        "scope": "scheduler-component audit with exact common recomposition; not full learned N02",
        "smoke": smoke,
        "scorer_seeds": [0] if smoke else [0, 1, 2],
        "trial_seeds": [26999] if smoke else list(range(27000, 27016)),
        "methods": list(METHODS),
        "patch_grid": [4, 4],
        "patch_size": 64,
        "frames": 8,
        "cue_nodes": [0, 5, 10, 15],
        "no_cue_nodes": "other twelve supports; not supplied to solver",
        "priors": "95% wrong on cue nodes; uniform on others; common explicit input",
        "dependency": "known task graph, random fixed one-hop directed edges p=.3; shared by all",
        "derived_state": "q_i = p_i * product(p_j for its recorded neighbours)",
        "scorer": "previous development-selected innovation CNN, frozen checkpoint200",
        "source_root": "outputs/n02-retrained-20261013/controls",
        "measurement": "streaming causal GRU, prefix score replaces evidence; IDs deduplicated",
        "absolute_fit": "measured-reference attenuation nuisance + fitted Gaussian positive cue; "
        "pooled RAW difference noise sqrt(2*.01^2/16), threshold2.5",
        "fit_limit": "flexible nonpositive nuisance; guard for unstructured bright corruption; "
        "not a calibrated full RAW density or hidden renderer oracle",
        "bad_measurement": "last frame node15 unstructured bright corruption in every fourth trial",
        "n02_rules": "original semantic_map, geometry gap and propagate at patch granularity",
        "common_recomposition": "exact algebraic state operator, same for global/local/cache; "
        "no learned cell/proposal errors in this conditional audit",
        "retention": "local and cached update direct factor states plus dependent readout states; "
        "standard/global recompute all supported factors and readouts",
        "drift": "derived q change outside actual cue/dependency affected set; evaluator-only mask",
        "cost": "full scorer+absolute-fit+controller+recomposition per frame, interleaved order",
        "unique_gain": "2/3 seeds: no correction degradation, and either drift improvement>=.01 "
        "or end-to-end latency ratio bootstrap95% upper<.9 versus event cached filter",
        "exit": "reliable scorer but no repeatable unique gain: archive current scheduler",
        "interpretation": "gain over dense alone is incremental computation, not uniqueness",
    }


def make_trial(seed):
    rng = np.random.default_rng(seed)
    cue_nodes = (0, 5, 10, 15)
    frames, truth, priors = [], [], []
    for node in range(16):
        world = int(rng.integers(0, 2))
        gen = SyntheticGenerator(
            SyntheticOptions(height=64, width=64), seed=seed * 16 + node, split="validation"
        )
        episode = gen.generate(
            seed=seed * 16 + node,
            episode_index=node,
            prior_condition="missing",
            identifiable=node in cue_nodes,
            value=("left", "right")[world],
        )
        frames.append(torch.from_numpy(episode.frames.copy()))
        truth.append(world)
        priors.append((0.05 if world else 0.95) if node in cue_nodes else 0.5)
    raw = torch.stack(frames)
    bad = seed % 4 == 3
    if bad:
        raw[15, -1] = 0.6 + torch.from_numpy(rng.normal(0, 0.1, (3, 64, 64))).float()
    valid = torch.zeros(4, 4, 4, dtype=torch.bool)
    neighbours = []
    for node in range(16):
        row, column = divmod(node, 4)
        peers = []
        for direction, (dy, dx) in enumerate(NEIGHBOURHOOD):
            y, x = row + dy, column + dx
            if 0 <= y < 4 and 0 <= x < 4 and rng.random() < 0.3:
                valid[row, column, direction] = True
                peers.append(4 * y + x)
        neighbours.append(peers)
    affected = torch.tensor(
        [
            node in cue_nodes or any(p in cue_nodes for p in peers)
            for node, peers in enumerate(neighbours)
        ]
    )
    return {
        "raw": raw,
        "truth": torch.tensor(truth),
        "prior": torch.tensor(priors),
        "valid": valid,
        "neighbours": neighbours,
        "affected": affected,
        "cue_nodes": cue_nodes,
        "bad_final": bad,
        "seed": seed,
    }


def absolute_patch_residual(frame, reference):
    """Fit each physical binary candidate from actual RAW, with no renderer state."""
    delta = F.avg_pool2d(frame - reference, 4)
    positive = delta.clamp_min(0)
    # The nonpositive difference is a fitted attenuation nuisance, identical for
    # both explanations. One nonnegative PSF describes the distinct cue marker.
    coordinate = torch.arange(3, dtype=frame.dtype, device=frame.device) - 1
    kernel = torch.exp(-(coordinate[:, None].square() + coordinate[None].square()) / (2 * 0.6**2))
    correlation = F.conv2d(positive.mean(1, keepdim=True), kernel[None, None], padding=1)[:, 0]
    weight = (reference.mean(1) - 0.05).clamp_min(0)
    yfull = torch.arange(64, device=frame.device, dtype=frame.dtype)[None, :, None]
    total = weight.sum((1, 2))
    center = (weight * yfull).sum((1, 2)) / total.clamp_min(1e-12)
    center = torch.where(total > 0, center / 4, torch.full_like(center, 8))
    y = torch.arange(16, device=frame.device, dtype=frame.dtype)[None, :, None]
    x = torch.arange(16, device=frame.device, dtype=frame.dtype)[None, None, :]
    residuals = []
    for world in (0, 1):
        permitted = (y >= center[:, None, None]) if world == 0 else (y < center[:, None, None])
        matched = correlation.masked_fill(~permitted.expand_as(correlation), -torch.inf)
        location = matched.flatten(1).argmax(1)
        amplitude = matched.flatten(1).max(1).values.clamp_min(0) / kernel.square().sum()
        ky, kx = location // 16, location % 16
        cue = (
            torch.exp(
                -((y - ky[:, None, None]).square() + (x - kx[:, None, None]).square())
                / (2 * 0.6**2)
            )
            * amplitude[:, None, None]
        )
        residual = (positive - cue[:, None]).abs().mean((1, 2, 3)) / math.sqrt(2 * 0.01**2 / 16)
        residuals.append(residual)
    return torch.stack(residuals, 1)


@dataclass
class MeasurementState:
    reference: torch.Tensor
    hidden: torch.Tensor
    last_values: torch.Tensor
    last_index: int
    scores: torch.Tensor
    residuals: torch.Tensor


@torch.no_grad()
def measurement_step(model, frame, state, index):
    if state is not None and index == state.last_index:
        if not torch.equal(frame, state.last_values):
            raise ValueError("same observation ID carries different measurements")
        return state
    if state is not None and index <= state.last_index:
        raise ValueError("observation ordering violated")
    reference = frame.clone() if state is None else state.reference
    raw = frame[:, None] - reference[:, None]
    standardized = ((raw - model.mean) / model.std).reshape(-1, 3, 64, 64)
    embedded = model.embedding(standardized)[:, None]
    previous = None if state is None else state.hidden
    output, hidden = model.temporal(embedded, previous)
    scores = model.classifier(output[:, 0]) - model.training_log_prior
    residual = absolute_patch_residual(frame, reference)
    return MeasurementState(reference, hidden, frame.clone(), index, scores, residual)


def evidence_maps(measurement):
    score, residual = measurement.scores, measurement.residuals
    best = score.argmax(1)
    best_residual = residual.gather(1, best[:, None])
    against = score.max(1).values[:, None] - score
    gap = (torch.arange(2)[None] != best[:, None]).float() * 2

    def maps(value):
        return value.reshape(4, 4, -1).permute(2, 0, 1)[None]

    return SimpleNamespace(
        support_mask=torch.ones(1, 2, 4, 4, dtype=torch.bool),
        local_log_evidence_against=maps(against),
        geometry_gap_from_best=maps(gap),
        best_normalized_residual=maps(best_residual).expand(1, 2, 4, 4),
    )


def compose_all(factors, neighbours):
    return torch.stack(
        [
            factors[i] * factors[peers].prod() if peers else factors[i]
            for i, peers in enumerate(neighbours)
        ]
    )


def compose_selected(factors, old, selection, neighbours):
    result = old.clone()
    for index in selection.nonzero().flatten().tolist():
        peers = neighbours[index]
        result[index] = factors[index] * factors[peers].prod() if peers else factors[index]
    return result


def cached_selection(direct, neighbours):
    return torch.tensor(
        [
            bool(direct[i]) or any(bool(direct[p]) for p in peers)
            for i, peers in enumerate(neighbours)
        ]
    )


def update_state(method, factors, derived, measurement, prior, controller, edge, valid, neighbours):
    energy = evidence_maps(measurement)
    allowed = (energy.best_normalized_residual[0, 0] <= 2.5).flatten()
    proposal = torch.sigmoid(
        torch.logit(prior.double()) + (measurement.scores[:, 1] - measurement.scores[:, 0]).double()
    ).float()
    if method in ("standard_filter", "global_update"):
        factors = torch.where(allowed, proposal, factors)
        if method == "standard_filter":
            derived = compose_all(factors, neighbours)
        else:
            weights = neighbour_stack(factors.reshape(4, 4))
            derived = (factors.reshape(4, 4) * torch.where(valid, weights, 1).prod(-1)).flatten()
        return factors, derived, 16
    contradiction = controller.semantic_map(energy)
    direct = contradiction.any((0, 1, 2)).flatten()
    if method == "n02_local":
        selection, _ = controller.propagate(contradiction, edge, valid[None, None])
        selection = selection.any((0, 1, 2)).flatten()
    else:
        selection = cached_selection(direct, neighbours)
    factors = torch.where(direct & allowed, proposal, factors)
    derived = compose_selected(factors, derived, selection, neighbours)
    return factors, derived, int(selection.sum())


def latency_interval(local, cached, rng):
    # Cluster at independent scene/trial, not at correlated frames.
    local, cached = np.array(local), np.array(cached)
    ratios = []
    for _ in range(2000):
        index = rng.integers(0, len(local), len(local))
        ratios.append(float(local[index].mean() / cached[index].mean()))
    return [float(np.quantile(ratios, q)) for q in (0.025, 0.975)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = protocol(args.smoke)
    source = Path(declaration["source_root"])
    assert json.loads((source / "summary.json").read_text())["acceptance"][
        "allow_mechanism_comparison"
    ]
    assert json.loads((source / "selection.json").read_text())["selected"] == "innovation"
    write_json(args.output / "protocol.json", declaration)
    trials = [make_trial(seed) for seed in declaration["trial_seeds"]]
    torch.save(trials, args.output / "trials.pt")
    source_paths = (
        [Path(__file__).resolve().relative_to(Path.cwd())]
        + sorted(Path("src/owac/frameworks/n02").glob("*.py"))
        + [
            Path("src/owac/baselines/n02_measurement.py"),
            Path("src/owac/baselines/n02_measurement_adapters.py"),
            Path("src/owac/baselines/sbi_embedding.py"),
        ]
    )
    hashes = {str(p): file_digest(p) for p in source_paths}
    manifest = {
        "source_sha256": hashes,
        "trial_raw_sha256": [tensor_digest(t["raw"]) for t in trials],
        "protocol_sha256": file_digest(args.output / "protocol.json"),
        "scorers": {},
    }
    for p in source_paths:
        destination = args.output / "source_snapshot" / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    stats = json.loads((source / "manifest.json").read_text())["modes"]["innovation"]
    config = N02Config.from_json(Path("configs/encoders/n02/tiny_contracts.json"))
    config.K = 2
    controller = RevisionController(config).eval()
    started = time.perf_counter()
    results = {}
    for seed in declaration["scorer_seeds"]:
        model = AdaptedMeasurementScorer(stats["mean"], stats["std"], "innovation").eval()
        checkpoint = source / "innovation" / f"seed{seed}" / "checkpoint200.pt"
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        )
        manifest["scorers"][str(seed)] = {
            "path": str(checkpoint),
            "sha256": file_digest(checkpoint),
        }
        rows = []
        for trial in trials:
            valid, neighbours = trial["valid"], trial["neighbours"]
            edge = valid[None, None, ..., None, None].float() * torch.eye(2).reshape(
                1, 1, 1, 1, 1, 2, 2
            )
            prior = trial["prior"]
            initial = compose_all(prior, neighbours)
            states = {method: (prior.clone(), initial.clone()) for method in METHODS}
            measurements = None
            timing = {method: [] for method in METHODS}
            records = []
            affected = trial["affected"]
            work = {method: 0 for method in METHODS}
            vetoes = {method: 0 for method in METHODS}
            drift = {method: [] for method in METHODS}
            for frame_index in range(8):
                frame = trial["raw"][:, frame_index]
                orders = METHODS[frame_index % 4 :] + METHODS[: frame_index % 4]
                shared = None
                for method in orders:
                    clock = time.perf_counter_ns()
                    measured = measurement_step(model, frame, measurements, frame_index)
                    factors, derived, count = update_state(
                        method,
                        *states[method],
                        measured,
                        prior,
                        controller,
                        edge,
                        valid,
                        neighbours,
                    )
                    timing[method].append((time.perf_counter_ns() - clock) / 1e6)
                    states[method] = (factors, derived)
                    work[method] += count
                    vetoes[method] += int((measured.residuals.min(1).values > 2.5).sum())
                    drift[method].append(
                        float((derived[~affected] - initial[~affected]).abs().mean())
                        if (~affected).any()
                        else 0.0
                    )
                    if shared is not None:
                        torch.testing.assert_close(measured.scores, shared.scores, rtol=0, atol=0)
                    shared = measured
                measurements = shared
                torch.testing.assert_close(states["standard_filter"][1], states["global_update"][1])
                torch.testing.assert_close(
                    states["n02_local"][0], states["event_cached_filter"][0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    states["n02_local"][1], states["event_cached_filter"][1], rtol=0, atol=0
                )
                records.append(
                    {
                        method: {
                            "factors": states[method][0].tolist(),
                            "derived": states[method][1].tolist(),
                        }
                        for method in METHODS
                    }
                )
            metrics = {}
            for method in METHODS:
                mass = torch.where(trial["truth"].bool(), states[method][0], 1 - states[method][0])
                metrics[method] = {
                    "corrected_cue_nodes": int((mass[list(trial["cue_nodes"])] >= 0.7).sum()),
                    "mean_unaffected_drift": float(np.mean(drift[method])),
                    "updated_readouts": work[method],
                    "absolute_vetoes": vetoes[method],
                    "latency_ms": timing[method],
                    "measurement_state_bytes": state_tensor_bytes(measurements),
                }
            rows.append(
                {
                    "trial_seed": trial["seed"],
                    "bad_final": trial["bad_final"],
                    "metrics": metrics,
                    "trajectory": records,
                }
            )
        write_json(args.output / f"seed{seed}.json", rows)
        summaries = {}
        for method in METHODS:
            timings = [v for row in rows for v in row["metrics"][method]["latency_ms"]]
            summaries[method] = {
                "corrected_nodes": sum(
                    row["metrics"][method]["corrected_cue_nodes"] for row in rows
                ),
                "denominator": 4 * len(rows),
                "mean_unaffected_drift": float(
                    np.mean([row["metrics"][method]["mean_unaffected_drift"] for row in rows])
                ),
                "updated_readouts": sum(row["metrics"][method]["updated_readouts"] for row in rows),
                "latency_p50_ms": float(np.quantile(timings, 0.5)),
                "latency_p95_ms": float(np.quantile(timings, 0.95)),
            }
        local = [np.mean(row["metrics"]["n02_local"]["latency_ms"]) for row in rows]
        cached = [np.mean(row["metrics"]["event_cached_filter"]["latency_ms"]) for row in rows]
        interval = latency_interval(local, cached, np.random.default_rng(29000 + seed))
        same_corrections = (
            summaries["n02_local"]["corrected_nodes"]
            >= summaries["event_cached_filter"]["corrected_nodes"]
        )
        drift_gain = (
            summaries["event_cached_filter"]["mean_unaffected_drift"]
            - summaries["n02_local"]["mean_unaffected_drift"]
        )
        results[str(seed)] = {
            "methods": summaries,
            "local_to_cached_latency_ratio_ci95": interval,
            "unique_gain": bool(same_corrections and (drift_gain >= 0.01 or interval[1] < 0.9)),
            "local_cached_states_bitwise_identical": True,
        }
    assert hashes == {str(p): file_digest(p) for p in source_paths}
    write_json(args.output / "manifest.json", manifest)
    summary = {
        "scope": declaration["scope"],
        "results": results,
        "unique_gain_seeds": sum(row["unique_gain"] for row in results.values()),
        "archive_current_scheduler": not args.smoke
        and sum(row["unique_gain"] for row in results.values()) < 2,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "source_unchanged": True,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
