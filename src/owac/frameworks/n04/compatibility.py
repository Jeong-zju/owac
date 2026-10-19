"""Implementation identity: parameter shapes alone cannot validate algorithm semantics."""

IMPLEMENTATION_VERSION = "n04-v1.1-s2-stability-repair-v5"
LOSS_DEFINITION = "v1_1_s2_normalized_latent_bounded_transition_v5"


def require_current_implementation(payload):
    if payload.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError(
            "checkpoint implementation_version is incompatible with this repair; "
            "use its frozen historical source for old results. A weight migration requires "
            "an explicit new experiment, not a silent resume."
        )
