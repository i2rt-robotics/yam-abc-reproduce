"""``yam-abc-deploy``: drive the YAM station from a policy server.

Reuses the same station config, robot units, and cameras as ``yam-abc-teleop``
-- the only difference is the per-tick command comes from a remote policy over
the unified websocket protocol instead of a teleop leader. Switching policies
(openpi / molmoact / abc) is just a different --host/--port; the client is
identical.

Because the policy replaces the leader, only the follower buses are opened: one
CAN channel per arm, and the teaching arms needn't be plugged in at all.

Example:
    yam-abc-deploy --station configs/station_yam.yaml \\
        --host 192.168.1.100 --port 8000 \\
        --prompt "pick up the red block" --seconds 60
"""

from __future__ import annotations

import argparse


def deploy(argv: list[str] | None = None) -> None:
    from ..config import build_station_config
    from ..runtime import build_arm_units, build_cameras_from_config
    from .client import WebsocketPolicyClient
    from .loop import DeployLoop

    p = argparse.ArgumentParser(prog="yam-abc-deploy")
    p.add_argument("--station", default="configs/station_yam.yaml")
    p.add_argument("--cameras", default=None)
    p.add_argument("--mock", action="store_true", help="use mock robot + cameras")
    p.add_argument("--host", required=True, help="policy server host/IP")
    p.add_argument("--port", type=int, required=True, help="policy server port")
    p.add_argument("--prompt", required=True, help="task instruction sent to the policy")
    p.add_argument("--seconds", type=float, default=60.0, help="rollout duration")
    p.add_argument(
        "--open-loop-horizon",
        type=int,
        default=15,
        help="rows to execute per predicted chunk before re-querying (= ABC execute_chunk_dim)",
    )
    p.add_argument("--rtc", action="store_true",
                   help="real-time chunking: async, prefix-conditioned (ABC/LBM style)")
    p.add_argument("--rtc-prefix-length", type=int, default=4, help="RTC: frozen prefix rows (P)")
    p.add_argument("--rtc-action-horizon", type=int, default=15, help="RTC: rows streamed per chunk (H)")
    p.add_argument("--rtc-lead-steps", type=int, default=4, help="RTC: re-query when L rows remain")
    p.add_argument("--ramp-seconds", type=float, default=1.0, help="ease-in to first action")
    p.add_argument(
        "--home-pose",
        default=None,
        help="comma/space-separated joint vector to ramp to before the policy takes over "
             "([joints..., gripper] per arm, in robots order). Defaults to the station's "
             "`deploy_home_pose`; pass an empty string to skip homing",
    )
    p.add_argument("--max-joint-speed", type=float, default=1.5,
                   help="max arm-joint speed in rad/s (safety clamp; <=0 disables)"
                        "(truncate + warn); 0 disables")
    p.add_argument(
        "--resize",
        default=None,
        help="optional HxW to resize images before send, e.g. 224x224 (default: full res)",
    )
    p.add_argument("--api-key", default=None)
    args = p.parse_args(argv)

    resize = None
    if args.resize:
        h, w = (int(x) for x in args.resize.lower().split("x"))
        resize = (h, w)

    cfg = build_station_config(args.station, args.cameras)

    # Home pose: the flag wins, else the station default. Validated up front so a wrong
    # length fails before the arms are built, not part-way through the ramp.
    if args.home_pose is None:
        home_pose = list(cfg.deploy_home_pose or [])
    else:
        home_pose = [float(x) for x in args.home_pose.replace(",", " ").split()]
    if home_pose:
        expected = len(cfg.robot.robots or [cfg.robot.active_robot()]) * (
            cfg.robot.num_arm_joints + 1
        )
        if len(home_pose) != expected:
            p.error(
                f"--home-pose has {len(home_pose)} values but this station needs "
                f"{expected} ([joints..., gripper] per arm, in robots order)"
            )

    # Autonomy commands the followers only, so no leader buses are opened.
    units = build_arm_units(cfg, mock=args.mock, followers_only=True)
    cameras = build_cameras_from_config(cfg, mock=args.mock)

    client = WebsocketPolicyClient(
        host=args.host,
        port=args.port,
        open_loop_horizon=args.open_loop_horizon,
        resize=resize,
        api_key=args.api_key,
        rtc=args.rtc,
        rtc_prefix_length=args.rtc_prefix_length,
        rtc_action_horizon=args.rtc_action_horizon,
        rtc_lead_steps=args.rtc_lead_steps,
    )
    print(f"connected to policy server {args.host}:{args.port}; metadata={client.metadata}")

    loop = DeployLoop(units, cameras, client, prompt=args.prompt, control_hz=cfg.control_hz,
                      max_joint_speed=args.max_joint_speed)
    try:
        loop.run_for(
            seconds=args.seconds, ramp_seconds=args.ramp_seconds, home_pose=home_pose
        )
    except KeyboardInterrupt:
        print("\ninterrupted -- estopping")
        loop.estop()


if __name__ == "__main__":
    deploy()
