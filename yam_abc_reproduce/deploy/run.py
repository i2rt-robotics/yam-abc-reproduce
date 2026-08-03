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

from dataclasses import dataclass

import tyro


@dataclass
class DeployArgs:
    host: str
    """policy server host/IP"""
    port: int
    """policy server port"""
    prompt: str
    """task instruction sent to the policy"""
    station: str = "configs/station_yam.yaml"
    cameras: str | None = None
    mock: bool = False
    """use mock robot + cameras"""
    seconds: float = 60.0
    """rollout duration"""
    open_loop_horizon: int = 15
    """rows to execute per predicted chunk before re-querying (= ABC execute_chunk_dim)"""
    rtc: bool = False
    """real-time chunking: async, prefix-conditioned (ABC/LBM style)"""
    rtc_prefix_length: int = 4
    """RTC: frozen prefix rows (P)"""
    rtc_action_horizon: int = 15
    """RTC: rows streamed per chunk (H)"""
    rtc_lead_steps: int = 4
    """RTC: re-query when L rows remain"""
    ramp_seconds: float = 1.0
    """ease-in to first action"""
    home_pose: str | None = None
    """comma/space-separated joint vector to ramp to before the policy takes over
    ([joints..., gripper] per arm, in robots order). Defaults to the station's
    `deploy_home_pose`; pass an empty string to skip homing"""
    max_joint_speed: float = 1.5
    """max arm-joint speed in rad/s (safety clamp; <=0 disables)(truncate + warn); 0 disables"""
    resize: str | None = None
    """optional HxW to resize images before send, e.g. 224x224 (default: full res)"""
    api_key: str | None = None


def deploy(argv: list[str] | None = None) -> None:
    from ..config import build_station_config
    from ..runtime import build_arm_units, build_cameras_from_config
    from .client import WebsocketPolicyClient
    from .loop import DeployLoop

    args = tyro.cli(DeployArgs, args=argv, prog="yam-abc-deploy")

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
            raise SystemExit(
                f"error: --home-pose has {len(home_pose)} values but this station needs "
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
