#!/usr/bin/env python
"""Calibrate a passive-GELLO leader: per-joint signs and the home zero.

Reading the GELLO is passive (multi-reader safe), so this can run alongside the GUI.
It never energizes the follower. Two steps:

  monitor  Live per-joint readout (raw rad + calibrated rad + gripper). Wiggle each
           GELLO joint to learn which index it is and which direction it counts.
  zero     Hold the GELLO at the pose you want mapped to the follower's home (all
           follower joints ~0) with the trigger released (it normalizes by distance from
           zero), then capture it. The zero is written into every encoder's EEPROM,
           gripper included (i2rt reset_zero_position), so there are no software offsets
           to persist. Re-run at a new pose to re-zero.

Signs (direction) are found empirically: with teleop running, move one joint; if the
follower goes the opposite way, flip that joint's sign in the station config's
`leader_joint_signs` (or, for one leader only, that controller's `joint_signs`). Use
`monitor` to confirm which index a joint is.

Usage:
  python scripts/calibrate_gello.py monitor --controller left
  python scripts/calibrate_gello.py zero    --controller left
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from yam_abc_reproduce.config import (
    ControllerConfig,
    build_station_config,
    controller_channel,
    gello_variant,
    is_passive_gello,
    leader_joint_signs_for,
)


def _controller_for_side(cfg, side: str) -> ControllerConfig:
    """The station's configured GELLO controller on that side (desk or mobile variant)."""
    for c in cfg.robot.controllers:
        if is_passive_gello(c.type) and c.type.endswith(side):
            return c
    return ControllerConfig(type=f"passive_gello_{side}")


def _leader(cfg, side: str):
    from yam_abc_reproduce.robot.passive_gello import PassiveGelloLeader

    controller = _controller_for_side(cfg, side)
    channel = controller_channel(controller.type)
    n = cfg.robot.num_arm_joints
    grip = cfg.robot.leader_gripper
    signs = leader_joint_signs_for(controller, cfg.robot.leader_joint_signs)
    gello_type, gello_side = gello_variant(controller.type)
    return (
        PassiveGelloLeader(
            channel=channel,
            num_arm_joints=n,
            gripper_config=tuple(grip) if grip else (n, 0.7, 0.0),
            joint_signs=signs,
            gello_type=gello_type,
            side=gello_side,
        ),
        controller.type,
        channel,
        n,
        signs,
    )


def cmd_monitor(cfg, side: str) -> None:
    lead, ctype, channel, n, resolved = _leader(cfg, side)
    signs = np.array(resolved or [1] * (n + 1), dtype=float)
    print(f"monitoring {ctype} on {channel} (Ctrl-C to stop). Columns: raw[rad] | calibrated[rad]")
    try:
        while True:
            raw, grip_raw, btns = lead.raw_angles()
            arm_cal, grip_norm, _ = lead.get_state()
            raw_s = " ".join(f"{v:+.3f}" for v in raw)
            cal_s = " ".join(f"{v:+.3f}" for v in arm_cal)
            print(
                f"  raw[{raw_s}] | cal[{cal_s}] | grip_raw={grip_raw:+.3f} "
                f"norm={grip_norm:.2f} btns={btns}   (signs={signs.astype(int).tolist()})",
                end="\r",
                flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        lead.stop()


def cmd_zero(cfg, side: str) -> None:
    """Hardware-zero every joint encoder at the current pose, gripper included; the zero is
    stored in encoder EEPROM, not the config. Hold the GELLO at the follower's home pose
    with the trigger released, since the trigger's zero is its released position."""
    lead, _, channel, _, _ = _leader(cfg, side)
    try:
        devs = lead.hardware_zero()
    finally:
        lead.stop()
    print(
        f"\nhardware-zeroed encoders {devs} on {channel} at the current pose "
        f"(arm joints + gripper).\n"
        "The zero is persisted in encoder EEPROM — no config change needed. If the "
        "trigger was NOT released, re-run with it released, or its normalized value "
        "will sit off-centre (and stick at one end once the offset exceeds the travel)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["monitor", "zero"])
    ap.add_argument("--controller", choices=["left", "right"], default="left")
    ap.add_argument("--station", default="configs/station_yam.yaml")
    ap.add_argument("--cameras", default="configs/cameras.yaml")
    args = ap.parse_args()

    cfg = build_station_config(args.station, args.cameras)
    if args.command == "monitor":
        cmd_monitor(cfg, args.controller)
    else:
        cmd_zero(cfg, args.controller)


if __name__ == "__main__":
    main()
