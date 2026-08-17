"""YAM-ABC-Reproduce command-line entry points."""

from __future__ import annotations

import argparse


def cameras(argv: list[str] | None = None) -> None:
    """List connected cameras (RealSense + V4L2/decxin) with serials for cameras.yaml."""
    argparse.ArgumentParser(prog="yam-abc-cameras").parse_args(argv)

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


def teleop(argv: list[str] | None = None) -> None:
    """Run teleop + (optional) recording from the command line, headless."""
    from .config import build_station_config
    from .runtime import build_arm_units, build_cameras_from_config
    from .teleop.loop import ControlLoop

    p = argparse.ArgumentParser(prog="yam-abc-teleop")
    p.add_argument("--station", default="configs/station_yam.yaml")
    p.add_argument("--cameras", default=None)
    p.add_argument("--mock", action="store_true", help="use mock robot + cameras")
    p.add_argument("--record", metavar="TASK", default=None, help="record one episode")
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args(argv)

    cfg = build_station_config(args.station, args.cameras)
    units = build_arm_units(cfg, mock=args.mock)
    cameras = build_cameras_from_config(cfg, mock=args.mock)
    loop = ControlLoop(units, cameras, control_hz=cfg.control_hz)
    loop.run_for(seconds=args.seconds, record_task=args.record, save_root=cfg.save_root,
                 station=cfg)


def convert(argv: list[str] | None = None) -> None:
    """Convert a default-format episode (or directory of them) to another format."""
    import os
    from pathlib import Path

    from .data.formats import convert_episode

    p = argparse.ArgumentParser(prog="yam-abc-convert")
    p.add_argument("src", help="episode dir (default format) or parent dir")
    p.add_argument("--to", default="lerobot", choices=["lerobot", "abc"])
    p.add_argument("--repo-id", default="yam_abc_reproduce/pick_and_place")
    p.add_argument("--out", default=None, help="output root (LeRobot dataset root)")
    args = p.parse_args(argv)
    if args.out is None and not os.environ.get("HF_LEROBOT_HOME"):
        os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).resolve().parents[1] / "data" / "lerobot")
    convert_episode(args.src, to=args.to, repo_id=args.repo_id, out=args.out)


def viz(argv: list[str] | None = None) -> None:
    """Visualize a converted LeRobot dataset (Reron viewer) for a sanity check."""
    from .data.visualize import visualize_lerobot

    p = argparse.ArgumentParser(prog="yam-abc-viz")
    p.add_argument("--repo-id", default="yam_abc_reproduce/pick_and_place")
    p.add_argument("--root", default=None)
    p.add_argument("--episode-index", type=int, default=0)
    args = p.parse_args(argv)
    visualize_lerobot(repo_id=args.repo_id, root=args.root, episode_index=args.episode_index)


def gui(argv: list[str] | None = None) -> None:
    """Launch the unified collect/train/deploy GUI."""
    import logging

    import uvicorn

    from .config import build_station_config
    from .gui.server import create_app

    p = argparse.ArgumentParser(prog="yam-abc-gui")
    p.add_argument("--station", default="configs/station_yam.yaml")
    p.add_argument("--cameras", default=None)
    p.add_argument("--mock", action="store_true", help="use mock robot + cameras")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8042)
    args = p.parse_args(argv)

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
