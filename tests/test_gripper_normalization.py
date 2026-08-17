"""Passive-GELLO trigger normalization.

A desk GELLO's trigger counts negative on squeeze and a mobile GELLO's counts positive,
so the mapping normalizes by distance from zero, as the fleet's own driver does. The
earlier signed mapping clipped a positive trigger's whole travel outside the range and
pinned it at 1.0, so mobile grippers silently stopped responding. Normalizing also makes
the gripper element of each side's joint_signs table inert.
"""

import pytest

from yam_abc_reproduce.robot.passive_gello import GRIPPER_RANGE, _normalize_gripper

CLOSED = -0.713  # measured full squeeze; only its magnitude is used


def test_released_trigger_is_fully_open():
    assert _normalize_gripper(0.0, CLOSED) == 1.0


@pytest.mark.parametrize("squeezed", [GRIPPER_RANGE, -GRIPPER_RANGE])
def test_full_squeeze_is_fully_closed_in_either_direction(squeezed):
    """The regression: a positive-counting trigger has to close too."""
    assert _normalize_gripper(squeezed, CLOSED) == 0.0


@pytest.mark.parametrize("angle", [0.05, 0.2, 0.33, 0.5, 0.67])
def test_the_curve_is_symmetric_about_zero(angle):
    assert _normalize_gripper(angle, CLOSED) == _normalize_gripper(-angle, CLOSED)


def test_it_sweeps_monotonically_between_the_ends():
    steps = 16
    seq = [_normalize_gripper(GRIPPER_RANGE * i / steps, CLOSED) for i in range(steps + 1)]
    assert seq[0] == 1.0
    assert seq[-1] == pytest.approx(0.0, abs=1e-9)
    assert seq[steps // 2] == pytest.approx(0.5)
    assert all(later < earlier for earlier, later in zip(seq, seq[1:]))


@pytest.mark.parametrize("overshoot", [1.5, -1.5, 100.0])
def test_past_the_end_clamps_to_closed(overshoot):
    """A negative value here would push the follower's gripper command past its range."""
    assert _normalize_gripper(overshoot, CLOSED) == 0.0


def test_travel_is_capped_at_the_reference_range():
    """|closed_rad| beyond GRIPPER_RANGE is capped, so a short squeeze still reads closed."""
    assert _normalize_gripper(GRIPPER_RANGE, -5.0) == 0.0


def test_a_shorter_measured_travel_is_used_as_is():
    """A trigger with less travel than GRIPPER_RANGE closes at its own measured end."""
    assert _normalize_gripper(0.3, -0.3) == 0.0
    assert _normalize_gripper(0.15, -0.3) == pytest.approx(0.5)


def test_a_degenerate_range_reports_closed_rather_than_dividing_by_zero():
    assert _normalize_gripper(0.0, 0.0) == 0.0


def test_an_unzeroed_trigger_sticks_at_one_end():
    """The mapping treats 0 as released, so an encoder zero offset past the travel pins the
    output. That is why ``hardware_zero`` must cover the gripper."""
    offset = 1.4  # released trigger reading +1.4 rad instead of 0
    at_rest = _normalize_gripper(offset, CLOSED)
    squeezed = _normalize_gripper(offset + GRIPPER_RANGE, CLOSED)
    assert at_rest == squeezed == 0.0


def test_hardware_zero_covers_every_joint_encoder_including_the_gripper():
    """The gripper used to be skipped, leaving its zero wherever the factory put it."""
    from yam_abc_reproduce.robot.passive_gello import PassiveGelloLeader

    zeroed: list[int] = []

    class _FakeEncoder:
        def reset_zero_position(self, device):
            zeroed.append(device)

    # Bypass __init__ so no CAN bus is opened; hardware_zero only needs these.
    lead = object.__new__(PassiveGelloLeader)
    lead._arm_devices = [0, 1, 2, 3, 4, 5]
    lead._grip_device = 6
    lead._encoder = _FakeEncoder()

    assert lead.hardware_zero() == [0, 1, 2, 3, 4, 5, 6]
    assert zeroed == [0, 1, 2, 3, 4, 5, 6]
