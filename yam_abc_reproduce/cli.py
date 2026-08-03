"""YAM-ABC-Reproduce command-line entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tyro


@dataclass
class CamerasArgs:
    pass


def cameras(argv: list[str] | None = None) -> None:
    """List connected cameras (RealSense + V4L2/decxin) with serials for cameras.yaml."""
    tyro.cli(CamerasArgs, args=argv, prog="yam-abc-cameras")

    from .camera.discovery import discover_realsense, discover_v4l2

    rs_devs = discover_realsense()
    print("RealSense devices:")
    if not rs_devs:
        print("  (none found)")
    for d in rs_devs:
        print(f"  {d['product']:<28} serial={d['serial']}   (type: realsense)")

    v4l2_devs = discover_v4l2()
    print("\nV4L2 / UVC devices (decxin, and RealSense as UVC):")
    if not v4l2_devs:
        print("  (none found)")
    for d in v4l2_devs:
        print(f"  {d['node']:<14} {d['product']:<28} serial={d['serial']}")

    print("\nPut a distinct `serial:` in configs/cameras.yaml for each camera")
    print("(required when several are the same model, e.g. multiple RealSense cameras).")
    print("Or configure them from live hardware in the GUI Station rail (Preview detects + saves).")


@dataclass
class TeleopArgs:
    station: str = "configs/station_yam.yaml"
    cameras: str | None = None
    mock: bool = False
    """use mock robot + cameras"""
    record: str | None = None
    """record one episode, under this TASK name"""
    seconds: float = 10.0


def teleop(argv: list[str] | None = None) -> None:
    """Run teleop + (optional) recording from the command line, headless."""
    from .config import build_station_config
    from .runtime import build_arm_units, build_cameras_from_config
    from .teleop.loop import ControlLoop

    args = tyro.cli(TeleopArgs, args=argv, prog="yam-abc-teleop")

    cfg = build_station_config(args.station, args.cameras)
    units = build_arm_units(cfg, mock=args.mock)
    cameras = build_cameras_from_config(cfg, mock=args.mock)
    loop = ControlLoop(units, cameras, control_hz=cfg.control_hz)
    loop.run_for(seconds=args.seconds, record_task=args.record, save_root=cfg.save_root,
                 station=cfg)


@dataclass
class ConvertArgs:
    src: tyro.conf.Positional[str]
    """episode dir (default format) or parent dir"""
    to: Literal["lerobot", "abc"] = "lerobot"
    repo_id: str = "yam_abc_reproduce/pick_and_place"
    out: str | None = None
    """output root (LeRobot dataset root)"""


def convert(argv: list[str] | None = None) -> None:
    """Convert a default-format episode (or directory of them) to another format."""
    import os
    from pathlib import Path

    from .data.formats import convert_episode

    args = tyro.cli(ConvertArgs, args=argv, prog="yam-abc-convert")
    if args.out is None and not os.environ.get("HF_LEROBOT_HOME"):
        os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).resolve().parents[1] / "data" / "lerobot")
    convert_episode(args.src, to=args.to, repo_id=args.repo_id, out=args.out)


@dataclass
class VizArgs:
    repo_id: str = "yam_abc_reproduce/pick_and_place"
    root: str | None = None
    episode_index: int = 0


def viz(argv: list[str] | None = None) -> None:
    """Visualize a converted LeRobot dataset (Reron viewer) for a sanity check."""
    from .data.visualize import visualize_lerobot

    args = tyro.cli(VizArgs, args=argv, prog="yam-abc-viz")
    visualize_lerobot(repo_id=args.repo_id, root=args.root, episode_index=args.episode_index)


@dataclass
class GuiArgs:
    station: str = "configs/station_yam.yaml"
    cameras: str | None = None
    mock: bool = False
    """use mock robot + cameras"""
    host: str = "0.0.0.0"
    port: int = 8042


def gui(argv: list[str] | None = None) -> None:
    """Launch the unified collect/train/deploy GUI."""
    import logging

    import uvicorn

    from .config import build_station_config
    from .gui.server import create_app

    args = tyro.cli(GuiArgs, args=argv, prog="yam-abc-gui")

    # Root logger at INFO so the station's own diagnostics (each follower's resolved gripper
    # travel, i2rt's gripper calibration) reach the terminal; the default is WARNING, no handler.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = build_station_config(args.station, args.cameras)
    app = create_app(cfg, mock=args.mock, station_path=args.station, cameras_path=args.cameras)

    # Silencing uvicorn (below) also drops its "running on ..." banner, so announce the URL
    # ourselves — on the startup event, which fires only once the port is bound.
    @app.on_event("startup")
    def _announce_url() -> None:
        import socket

        local = "localhost" if args.host in ("0.0.0.0", "::", "") else args.host
        logging.info("GUI ready — open http://%s:%d in your browser", local, args.port)
        if args.host in ("0.0.0.0", "::"):
            # Bound to every interface, so a laptop driving this station can reach it too.
            logging.info("  from another machine: http://%s:%d", socket.gethostname(), args.port)

    # The access log buries everything else: the GUI re-fetches each camera preview 5x a second.
    # Uvicorn's dictConfig leaves our root handler alone (disable_existing_loggers=False).
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="warning")
