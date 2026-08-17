"""ZMQ-based camera server.

Hosts the RealSense cameras in a long-lived process so eval clients can pull
the latest frames on demand without paying for pipeline startup, fighting the
policy loop for camera I/O, or coupling robot-control timing to camera I/O.

Sockets
-------
REP  ``tcp://127.0.0.1:5555``  (default)
    Pull semantics. Client sends a pickled request dict; server replies with a
    pickled response dict. Used by the policy for on-demand obs.

PUB  ``tcp://127.0.0.1:5556``  (default, optional)
    Push semantics. Server publishes the latest obs every ``pub_period_sec``.
    Intended for the cv2 live viewer so it can render at camera rate without
    burning policy-side requests.

Request protocol
----------------
    {"cmd": "obs"}   ->  {"ok": True, "frames": {cam_name: np.ndarray (H,W,3) uint8 RGB},
                          "timestamps": {cam_name: float}}
    {"cmd": "ping"}  ->  {"ok": True, "pong": True}

Errors come back as ``{"ok": False, "error": str}``. The server keeps running
across any single bad request.

CLI
---
    python examples/yam/camera_server.py --config examples/yam/configs/yam_left.yaml
"""
from __future__ import annotations

import argparse
import logging
import pickle
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import zmq
from omegaconf import OmegaConf

from gello_min.realsense_camera import RealSenseCamera, get_device_ids


logger = logging.getLogger("camera_server")


DEFAULT_REP_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_PUB_ENDPOINT = "tcp://127.0.0.1:5556"
DEFAULT_PUB_PERIOD_SEC = 1.0 / 30.0
DEFAULT_HEARTBEAT_SEC = 10.0


class CameraServer:
    """Owns RealSense cameras and serves their latest frames over ZMQ."""

    def __init__(
        self,
        cameras: Dict[str, RealSenseCamera],
        rep_endpoint: str = DEFAULT_REP_ENDPOINT,
        pub_endpoint: Optional[str] = None,
        pub_period_sec: float = DEFAULT_PUB_PERIOD_SEC,
        heartbeat_sec: float = DEFAULT_HEARTBEAT_SEC,
    ) -> None:
        self.cameras = cameras
        self.rep_endpoint = rep_endpoint
        self.pub_endpoint = pub_endpoint
        self.pub_period_sec = float(pub_period_sec)
        self.heartbeat_sec = float(heartbeat_sec)

        self._ctx = zmq.Context.instance()
        self._rep: Optional[zmq.Socket] = None
        self._pub: Optional[zmq.Socket] = None

        self._stop_event = threading.Event()
        self._pub_thread: Optional[threading.Thread] = None

        self._req_total = 0
        self._req_window = 0
        self._last_heartbeat = time.time()

    # ------------------------------------------------------------------
    # Frame sourcing
    # ------------------------------------------------------------------

    def _snapshot(self) -> Dict[str, Any]:
        """Snapshot the latest color frame from every camera (RGB uint8)."""
        frames: Dict[str, Any] = {}
        timestamps: Dict[str, float] = {}
        for name, cam in self.cameras.items():
            image, _depth = cam.read()
            frames[name] = image
            # Surface the capture timestamp so the client can detect staleness.
            ts = getattr(cam, "_latest_frame_timestamp", None) or 0.0
            timestamps[name] = float(ts)
        return {"ok": True, "frames": frames, "timestamps": timestamps}

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def _handle_request(self) -> None:
        assert self._rep is not None
        raw = self._rep.recv()
        try:
            req = pickle.loads(raw)
            cmd = (req or {}).get("cmd", "obs")
        except Exception as exc:  # noqa: BLE001 — surface to client, stay alive
            self._rep.send(pickle.dumps({"ok": False, "error": f"bad request: {exc!r}"}))
            return

        try:
            if cmd == "obs":
                resp = self._snapshot()
            elif cmd == "ping":
                resp = {"ok": True, "pong": True}
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except Exception as exc:  # noqa: BLE001 — keep server alive
            logger.exception("Request failed (cmd=%r)", cmd)
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        self._rep.send(pickle.dumps(resp), copy=False)
        self._req_total += 1
        self._req_window += 1

    def _pub_loop(self) -> None:
        assert self._pub is not None
        next_tick = time.time()
        while not self._stop_event.is_set():
            now = time.time()
            if now < next_tick:
                # Tiny sleep granularity so shutdown is snappy.
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + self.pub_period_sec
            try:
                resp = self._snapshot()
                self._pub.send(pickle.dumps(resp), copy=False)
            except Exception as exc:  # noqa: BLE001 — pub is best-effort
                logger.warning("PUB tick failed: %s", exc)

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        elapsed = now - self._last_heartbeat
        if elapsed < self.heartbeat_sec:
            return
        hz = self._req_window / elapsed if elapsed > 0 else 0.0
        logger.info(
            "alive: total_requests=%d window=%d (%.1f req/s) cameras=%d",
            self._req_total, self._req_window, hz, len(self.cameras),
        )
        self._req_window = 0
        self._last_heartbeat = now

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(self.rep_endpoint)
        logger.info("REP bound on %s", self.rep_endpoint)

        if self.pub_endpoint:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.bind(self.pub_endpoint)
            logger.info(
                "PUB bound on %s (period=%.3fs)", self.pub_endpoint, self.pub_period_sec,
            )
            self._pub_thread = threading.Thread(
                target=self._pub_loop, name="camera_server_pub", daemon=True,
            )
            self._pub_thread.start()

        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        try:
            while not self._stop_event.is_set():
                # 100 ms tick keeps heartbeats responsive and shutdown snappy.
                socks = dict(poller.poll(timeout=100))
                if self._rep in socks:
                    self._handle_request()
                self._maybe_heartbeat()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        if self._pub_thread is not None:
            self._pub_thread.join(timeout=2.0)
        for sock in (self._rep, self._pub):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
        for cam in self.cameras.values():
            try:
                cam._stop_event.set()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        logger.info("Camera server stopped.")


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def _build_cameras_from_config(cfg_path: Path) -> Dict[str, RealSenseCamera]:
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    camera_cfg = cfg["sensors"]["cameras"]
    logger.info("Discovering RealSense devices...")
    ids = get_device_ids()
    logger.info("Found %d RealSense devices: %s", len(ids), ids)
    cameras: Dict[str, RealSenseCamera] = {}
    for name, spec in camera_cfg.items():
        device_id = spec["device_id"]
        logger.info("Opening camera %s (device_id=%s)", name, device_id)
        cameras[name] = RealSenseCamera(device_id)
    return cameras


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="YAM camera server (ZMQ).")
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to a yam_*.yaml whose sensors.cameras block lists the devices.",
    )
    parser.add_argument("--rep-endpoint", default=DEFAULT_REP_ENDPOINT)
    parser.add_argument(
        "--pub-endpoint", default=DEFAULT_PUB_ENDPOINT,
        help="ZMQ PUB endpoint. Pass empty string to disable the PUB stream.",
    )
    parser.add_argument("--pub-period-sec", type=float, default=DEFAULT_PUB_PERIOD_SEC)
    parser.add_argument("--heartbeat-sec", type=float, default=DEFAULT_HEARTBEAT_SEC)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cameras = _build_cameras_from_config(args.config)
    server = CameraServer(
        cameras=cameras,
        rep_endpoint=args.rep_endpoint,
        pub_endpoint=(args.pub_endpoint or None),
        pub_period_sec=args.pub_period_sec,
        heartbeat_sec=args.heartbeat_sec,
    )

    def _handle(signum, _frame):
        logger.info("Signal %d received; shutting down.", signum)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
