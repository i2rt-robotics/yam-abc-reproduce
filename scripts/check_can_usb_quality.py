#!/usr/bin/env python3
"""Read-only SocketCAN and USB quality monitor for the four YAM CAN adapters."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_IFACES = ["can_left", "can_right", "can_lead_l", "can_lead_r"]


def link_snapshot(iface: str) -> dict:
    """Read kernel SocketCAN counters; this never opens or writes a CAN socket."""
    proc = subprocess.run(
        ["ip", "-j", "-statistics", "-details", "link", "show", iface],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        return {"interface": iface, "error": proc.stderr.strip() or "interface not found"}
    data = json.loads(proc.stdout)[0]
    info = data.get("linkinfo", {}).get("info_data", {})
    stats = data.get("stats64", {})
    xstats = data.get("info_xstats", {})
    usb_link = Path(f"/sys/class/net/{iface}/device")
    try:
        usb_path = str(usb_link.resolve())
    except OSError:
        usb_path = None
    return {
        "interface": iface,
        "state": info.get("state", data.get("operstate")),
        "flags": data.get("flags", []),
        "bitrate": info.get("bitrate"),
        "parentbus": data.get("parentbus"),
        "parentdev": data.get("parentdev"),
        "usb_sysfs": usb_path,
        "rx": stats.get("rx", {}),
        "tx": stats.get("tx", {}),
        "can_errors": xstats,
    }


def counter_delta(current: dict, previous: dict, path: tuple[str, ...]) -> int:
    cur = current
    prev = previous
    for key in path:
        cur = cur.get(key, 0) if isinstance(cur, dict) else 0
        prev = prev.get(key, 0) if isinstance(prev, dict) else 0
    return max(0, int(cur or 0) - int(prev or 0))


def quality_flags(current: dict, previous: dict | None) -> list[str]:
    flags: list[str] = []
    if current.get("error"):
        return ["NOT_FOUND"]
    if current.get("state") != "ERROR-ACTIVE":
        flags.append(f"CAN_STATE={current.get('state')}")
    if previous is None:
        return flags
    watched = [
        ("rx.errors", ("rx", "errors")),
        ("rx.dropped", ("rx", "dropped")),
        ("rx.over_errors", ("rx", "over_errors")),
        ("tx.errors", ("tx", "errors")),
        ("bus_error", ("can_errors", "bus_error")),
        ("arbitration_lost", ("can_errors", "arbitration_lost")),
        ("error_warning", ("can_errors", "error_warning")),
        ("error_passive", ("can_errors", "error_passive")),
        ("bus_off", ("can_errors", "bus_off")),
        ("restarts", ("can_errors", "restarts")),
    ]
    for label, path in watched:
        delta = counter_delta(current, previous, path)
        if delta:
            flags.append(f"{label}+{delta}")
    return flags


def print_sample(index: int, now: str, samples: list[dict], previous: list[dict] | None) -> None:
    print(f"[{now}] sample {index}")
    old_by_name = {x.get("interface"): x for x in previous or []}
    for item in samples:
        old = old_by_name.get(item.get("interface"))
        flags = quality_flags(item, old)
        if item.get("error"):
            print(f"  {item['interface']:<12} ERROR {item['error']}")
            continue
        rx_rate = counter_delta(item, old or {}, ("rx", "packets"))
        tx_rate = counter_delta(item, old or {}, ("tx", "packets"))
        state = item.get("state", "unknown")
        message = f"  {item['interface']:<12} {state:<13} RX+{rx_rate:<5} TX+{tx_rate:<5}"
        if flags:
            message += "  WARNING: " + ", ".join(flags)
        print(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only CAN/USB transport monitor. It never transmits CAN frames."
    )
    parser.add_argument("--seconds", type=float, default=60.0, help="total sampling time")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    parser.add_argument("--interfaces", nargs="+", default=DEFAULT_IFACES)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("--seconds and --interval must be positive")

    print("CAN/USB quality monitor (read-only)")
    print("Interfaces:", ", ".join(args.interfaces))
    print("Tip: move both leaders normally for 10 s, then reproduce the fast movement once.")

    report = {"started_at": datetime.now().isoformat(timespec="seconds"), "samples": []}
    previous: list[dict] | None = None
    deadline = time.monotonic() + args.seconds
    index = 0
    while True:
        now = datetime.now().isoformat(timespec="seconds")
        samples = [link_snapshot(iface) for iface in args.interfaces]
        report["samples"].append({"time": now, "interfaces": samples})
        print_sample(index, now, samples, previous)
        previous = samples
        index += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report: {args.out}")

    warning_count = 0
    for i in range(1, len(report["samples"])):
        old = {x.get("interface"): x for x in report["samples"][i - 1]["interfaces"]}
        for item in report["samples"][i]["interfaces"]:
            warning_count += len(quality_flags(item, old.get(item.get("interface"))))
    print(f"\nDone. Kernel-visible warnings: {warning_count}")
    print("Note: zero SocketCAN errors does not prove zero USB micro-latency; compare reports while reproducing the jerk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
