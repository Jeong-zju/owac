from copy import deepcopy

from owac.runtime.robotwin_grasp_acceptance import GraspHold


def sample(tick, *, table=False, hands=True, z=0.86):
    return {
        "physics_tick": tick,
        "dt": 0.02,
        "roller_p": [0, 0, z],
        "gripper_link_names": ["fl_link7", "fr_link7"],
        "closed_flags": [True, True],
        "roller_contacts": (
            [
                {"bodies": ["102_roller", "fl_link7"], "points": [{}]},
                {"bodies": ["102_roller", "fr_link7"], "points": [{}]},
            ]
            if hands
            else []
        )
        + ([{"bodies": ["102_roller", "table"], "points": [{}]}] if table else []),
    }


def test_height_and_closed_commands_do_not_prove_grasp():
    check = GraspHold()
    for t in range(100):
        assert not check.update(sample(t, table=True, hands=False, z=0.801))
    assert check.max_hold_seconds == 0


def test_grasp_must_be_sustained_and_resets_after_loss():
    check = GraspHold()
    for t in range(40):
        assert not check.update(sample(t))
    assert not check.update(sample(40, hands=False))
    assert not check.update(sample(41))
    assert check.update(sample(92)) is False  # long gaps cannot prove continuous hold
    for t in range(93, 144):
        accepted = check.update(sample(t))
    assert accepted
    bad = deepcopy(sample(144))
    bad["roller_p"][2] = 0.75
    assert not check.update(bad)
