#!/usr/bin/env python3
"""Passive SocketCAN reply-cadence monitor for the YAM-ABC-Reproduce YAM station.

This tool only receives CAN frames.  It never transmits a CAN command, so it
is safe to keep running while normal teleoperation is active.
"""

from __future__ import annotations

import json
import math
import select
import socket
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import tyro


CAN_FRAME = struct.Struct("=IB3x8s")
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF

FOLLOWER_REPLY_IDS = {0x10 + motor_id: motor_id for motor_id in range(1, 7)}
GELLO_REPORT_ID = 0x50F
DEFAULT_INTERFACES = ("can_lead_l", "can_lead_r", "can_left", "can_right")


@dataclass
class StreamStats:
    label: str
    intervals_ms: list[float] = field(default_factory=list)
    late_intervals_ms: list[float] = field(default_factory=list)
    last_seen: float | None = None
    frames: int = 0

    def observe(self, now: float) -> None:
        self.frames += 1
        if self.last_seen is not None:
            self.intervals_ms.append((now - self.last_seen) * 1000.0)
        self.last_seen = now

    def summary(self) -> dict[str, float | int | None]:
        values = sorted(self.intervals_ms)
        if not values:
            return {
                "frames": self.frames,
                "samples": 0,
                "mean_ms": None,
                "p95_ms": None,
                "max_ms": None,
                "late_count": 0,
            }
        mean = sum(values) / len(values)
        p95 = percentile(values, 95)
        threshold = max(mean * 2.5, p95 * 1.35, 20.0)
        late = [value for value in values if value > threshold]
        self.late_intervals_ms = late
        return {
            "frames": self.frames,
            "samples": len(values),
            "mean_ms": round(mean, 3),
            "p95_ms": round(p95, 3),
            "max_ms": round(values[-1], 3),
            "threshold_ms": round(threshold, 3),
            "late_count": len(late),
            "late_ratio_percent": round(len(late) * 100.0 / len(values), 3),
        }


def percentile(values: list[float], percent: int) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * percent / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def label_frame(interface: str, can_id: int, data: bytes) -> str | None:
    """Return a stable per-device label for known YAM reply frames."""
    if interface in {"can_left", "can_right"} and can_id in FOLLOWER_REPLY_IDS:
        return f"{interface}: motor_{FOLLOWER_REPLY_IDS[can_id]} (reply 0x{can_id:03X})"
    if interface in {"can_lead_l", "can_lead_r"} and can_id == GELLO_REPORT_ID and data:
        return f"{interface}: encoder_{data[0]} (report 0x{can_id:03X})"
    return None


def open_can_socket(interface: str) -> socket.socket:
    can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    can_socket.bind((interface,))
    can_socket.setblocking(False)
    return can_socket


@dataclass
class MonitorCanReplyLatencyArgs:
    seconds: float = 60.0
    """capture duration"""
    interfaces: list[str] = field(default_factory=lambda: list(DEFAULT_INTERFACES))
    """SocketCAN interfaces to monitor"""
    out: Path | None = None
    """optional JSON result file"""
    include_unknown: bool = False
    """also report unrecognised CAN IDs (useful for diagnosing a custom bus layout)"""


def main() -> None:
    args = tyro.cli(
        MonitorCanReplyLatencyArgs,
        description="Passively measure YAM CAN reply cadence during teleoperation.",
    )

    sockets: dict[socket.socket, str] = {}
    for interface in args.interfaces:
        try:
            can_socket = open_can_socket(interface)
        except OSError as exc:
            print(f"WARN  {interface}: cannot open ({exc})")
            continue
        sockets[can_socket] = interface
        print(f"LISTEN {interface}")

    if not sockets:
        raise SystemExit("No CAN interfaces could be opened. Check that SocketCAN is UP.")

    streams: dict[str, StreamStats] = {}
    started = time.monotonic()
    deadline = started + args.seconds
    print(f"Monitoring for {args.seconds:.1f}s. Start normal teleoperation now; this tool sends no CAN frames.")

    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select(list(sockets), [], [], 0.25)
            for can_socket in readable:
                try:
                    frame = can_socket.recv(CAN_FRAME.size)
                except BlockingIOError:
                    continue
                if len(frame) != CAN_FRAME.size:
                    continue
                raw_id, dlc, payload = CAN_FRAME.unpack(frame)
                if raw_id & (CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_ERR_FLAG):
                    continue
                can_id = raw_id & CAN_SFF_MASK
                interface = sockets[can_socket]
                label = label_frame(interface, can_id, payload[:dlc])
                if label is None:
                    if not args.include_unknown:
                        continue
                    label = f"{interface}: raw_0x{can_id:03X}"
                stats = streams.setdefault(label, StreamStats(label))
                stats.observe(time.monotonic())
    except KeyboardInterrupt:
        print("Stopped early by user.")
    finally:
        for can_socket in sockets:
            can_socket.close()

    report = {label: stats.summary() for label, stats in sorted(streams.items())}
    ranked = sorted(
        report.items(),
        key=lambda item: (
            item[1]["late_ratio_percent"] if item[1]["late_ratio_percent"] is not None else -1,
            item[1]["max_ms"] if item[1]["max_ms"] is not None else -1,
        ),
        reverse=True,
    )

    print("\nRESULTS (highest late-frame ratio first)")
    if not ranked:
        print("No known reply frames were observed. Ensure teleoperation is active and use --include-unknown if needed.")
    for label, result in ranked:
        if result["samples"] == 0:
            print(f"{label}: {result['frames']} frame(s), not enough samples")
            continue
        flag = "  <-- investigate" if result["late_count"] else ""
        print(
            f"{label}: avg {result['mean_ms']:.2f} ms, p95 {result['p95_ms']:.2f} ms, "
            f"max {result['max_ms']:.2f} ms, late {result['late_count']}/{result['samples']} "
            f"(>{result['threshold_ms']:.2f} ms){flag}"
        )

    payload = {
        "duration_seconds": round(time.monotonic() - started, 3),
        "interfaces": args.interfaces,
        "note": "Passive monitor. Measurements are arrival gaps between reply frames, not injected request-response probes.",
        "streams": report,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {args.out}")


if __name__ == "__main__":
    main()
