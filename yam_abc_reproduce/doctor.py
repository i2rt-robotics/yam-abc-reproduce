"""yam-abc-doctor: read-only station self-check.

Compares what the hardware presents (CAN interfaces and their traffic fingerprints,
cameras, GPU, system deps) against what the station config expects, printing PASS/WARN/FAIL
rows with a concrete fix for each finding. Entirely passive — no CAN frame sent, no camera
streamed, no process touched — so it is safe to run even while teleop is live.

Usage:
    yam-abc-doctor                 # checks against configs/station_yam.yaml
    yam-abc-doctor --config path/to/station.yaml
    yam-abc-doctor --no-listen    # skip the 1s-per-bus passive CAN listen
    yam-abc-doctor --json         # machine-readable output

Exit code: 0 = no FAIL rows, 1 = at least one FAIL, 2 = doctor itself errored.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# --- CAN protocol fingerprints (matches scripts/monitor_can_reply_latency.py) ---
_CAN_FRAME = struct.Struct("=IB3x8s")
_CAN_SFF_MASK = 0x000007FF
GELLO_REPORT_ID = 0x50F                      # passive-GELLO encoder broadcast
MOTOR_REPLY_IDS = {0x10 + m for m in range(1, 8)}   # DM motor replies 0x11..0x17
MOTOR_CMD_IDS = set(range(0x01, 0x08))       # DM motor command ids 0x01..0x07

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Finding:
    check: str
    status: str
    detail: str
    fix: str = ""


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_station(config_path: Path) -> dict:
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("robot", {}) | {"cameras_config": raw.get("cameras_config"),
                                   "_root": raw}


def expected_channels(robot_cfg: dict) -> dict[str, str]:
    """{channel_name: kind} from the station config, where kind is 'follower' (motor
    chain) or the controller type prefix ('passive_gello' / 'yam_lead')."""
    from .config import controller_channel, is_passive_gello, robot_channel

    def _explicit(entry: dict) -> str | None:
        """Operator-assigned channel override, as in robot_channel_for / controller_channel_for."""
        return (entry.get("channel") or "").strip() or None

    out: dict[str, str] = {}
    for r in robot_cfg.get("robots") or []:
        out[_explicit(r) or robot_channel(r["type"])] = "follower"
    for c in robot_cfg.get("controllers") or []:
        kind = "passive_gello" if is_passive_gello(str(c["type"])) else "yam_lead"
        out[_explicit(c) or controller_channel(c["type"])] = kind
    return out


# ---------------------------------------------------------------------------
# CAN checks
# ---------------------------------------------------------------------------

def can_interfaces() -> set[str]:
    root = Path("/sys/class/net")
    return {p.name for p in root.iterdir() if p.name.startswith("can")} if root.is_dir() else set()


def listen_can(iface: str, seconds: float = 1.0) -> set[int]:
    """Passively collect CAN ids seen on ``iface`` for ``seconds`` (RX only)."""
    ids: set[int] = set()
    try:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((iface,))
        s.setblocking(False)
    except OSError:
        return ids
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            r, _, _ = select.select([s], [], [], 0.1)
            if not r:
                continue
            try:
                frame = s.recv(16)
            except OSError:
                break
            can_id, _, _ = _CAN_FRAME.unpack(frame)
            ids.add(can_id & _CAN_SFF_MASK)
    finally:
        s.close()
    return ids


def classify_traffic(ids: set[int]) -> str:
    """'gello' | 'motor_chain' | 'silent' | 'other' from a passive id sample."""
    if not ids:
        return "silent"
    if GELLO_REPORT_ID in ids:
        return "gello"
    if ids & MOTOR_REPLY_IDS or ids & MOTOR_CMD_IDS:
        return "motor_chain"
    return "other"


def check_can(expected: dict[str, str], do_listen: bool) -> list[Finding]:
    out: list[Finding] = []
    present = can_interfaces()
    unnamed = sorted(i for i in present if re.fullmatch(r"can\d+", i))
    if unnamed:
        out.append(Finding(
            "can.udev", WARN,
            f"unnamed adapters present: {', '.join(unnamed)}",
            "run: python scripts/setup_can_udev.py (writes /etc/udev/rules.d/90-can.rules)"))
    for ch, kind in sorted(expected.items()):
        if ch not in present:
            out.append(Finding(
                f"can.{ch}", FAIL,
                f"interface missing (config expects it for a {kind})",
                "check the adapter's USB cable, then scripts/setup_can_udev.py for naming"))
            continue
        if not do_listen:
            out.append(Finding(f"can.{ch}", PASS, "interface present (listen skipped)"))
            continue
        kindseen = classify_traffic(listen_can(ch))
        if kindseen == "silent":
            out.append(Finding(
                f"can.{ch}", PASS,
                "present; no traffic (normal for idle/unpowered motors)"))
        elif kind == "passive_gello" and kindseen == "motor_chain":
            out.append(Finding(
                f"can.{ch}", FAIL,
                "config says passive GELLO but the bus carries a MOTOR chain",
                "leader type mismatch: set controllers to yam_lead_* in configs/station_yam.yaml, "
                "or the cables are swapped"))
        elif kind in ("yam_lead", "follower") and kindseen == "gello":
            out.append(Finding(
                f"can.{ch}", FAIL,
                "config expects motors but the bus carries a GELLO encoder broadcast",
                "leader type mismatch or swapped cables: check controllers in "
                "configs/station_yam.yaml / re-run scripts/setup_can_udev.py"))
        else:
            out.append(Finding(f"can.{ch}", PASS, f"present; traffic looks like {kindseen}"))
    return out


# ---------------------------------------------------------------------------
# camera / gpu / system checks
# ---------------------------------------------------------------------------

def check_cameras(repo_root: Path, cameras_config: str | None) -> list[Finding]:
    cams_file = repo_root / (cameras_config or "configs/cameras.yaml")
    if not cams_file.is_file():
        return [Finding("cameras.config", WARN, f"{cams_file.name} not found",
                        "the GUI Preview button can detect + write it")]
    import yaml

    cams = (yaml.safe_load(cams_file.read_text()) or {}).get("cameras", [])
    expected = {c["serial"]: c["name"] for c in cams if c.get("type") == "realsense" and c.get("serial")}
    if not expected:
        return [Finding("cameras", SKIP, "no realsense serials configured")]
    try:
        import pyrealsense2 as rs  # type: ignore
    except ImportError:
        return [Finding("cameras", SKIP, "pyrealsense2 not installed",
                        "uv sync --extra camera")]
    present = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
    out = []
    for serial, name in expected.items():
        if serial in present:
            out.append(Finding(f"cameras.{name}", PASS, f"detected (SN {serial})"))
        else:
            out.append(Finding(f"cameras.{name}", FAIL, f"SN {serial} not on USB",
                               "check the cable; if it streams nowhere, use the GUI camera "
                               "panel's Reset Cameras"))
    extra = present - set(expected)
    if extra:
        out.append(Finding("cameras.extra", WARN, f"unconfigured devices: {sorted(extra)}",
                           "add them in the GUI Station rail or configs/cameras.yaml"))
    return out


def check_gpu() -> list[Finding]:
    if not shutil.which("nvidia-smi"):
        return [Finding("gpu", SKIP, "nvidia-smi not found (robot-host-only machine?)")]
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        free, total, util = [int(x) for x in q.stdout.strip().splitlines()[0].split(",")]
    except Exception as e:  # noqa: BLE001
        return [Finding("gpu", WARN, f"nvidia-smi failed: {e}")]
    st = PASS if free > 8000 else WARN
    fix = "" if st == PASS else "free VRAM low — check squatters via the GUI GPU badge"
    return [Finding("gpu", st, f"{free}/{total} MiB free, util {util}%", fix)]


def check_system(repo_root: Path, expected_ch: dict[str, str]) -> list[Finding]:
    out = []
    # Recording and Review go through PyAV now, so a missing system ffmpeg no longer blanks
    # previews. It is still what abc's export_mcap.py shells out to when building the
    # training cache.
    out.append(Finding("sys.ffmpeg", PASS, "found") if shutil.which("ffmpeg") else
               Finding("sys.ffmpeg", WARN, "not found — the ABC training-cache build needs it",
                       "sudo apt install ffmpeg"))
    # /etc/sudoers.d is often 0750 root:root — stat may be denied for normal users.
    # sudo -n is the authoritative, permission-free probe: can we bring a CAN
    # interface up without a password prompt?
    probe = subprocess.run(
        ["sudo", "-n", "ip", "link", "set", "can_doctor_probe", "down"],
        capture_output=True, text=True)
    err = (probe.stderr or "").lower()
    if "password is required" in err or "a password is required" in err:
        out.append(Finding("sys.sudoers", FAIL, "no passwordless CAN rule",
                           "sudo bash scripts/setup_can_sudoers.sh"))
    elif probe.returncode != 0 and "cannot find device" in err:
        out.append(Finding("sys.sudoers", PASS,
                           "passwordless CAN rule works (probe device absent, as expected)"))
    elif probe.returncode == 0:
        out.append(Finding("sys.sudoers", PASS, "passwordless CAN rule works"))
    else:
        out.append(Finding("sys.sudoers", WARN,
                           f"could not determine (sudo -n said: {err.strip()[:80]})",
                           "verify manually: sudo -n ip link set can_left down"))
    rules = Path("/etc/udev/rules.d/90-can.rules")
    named_needed = any(not re.fullmatch(r"can\d+", c) for c in expected_ch)
    if named_needed and not rules.exists():
        out.append(Finding("sys.udev", WARN, "90-can.rules missing (names rely on plug order)",
                           "python scripts/setup_can_udev.py"))
    else:
        out.append(Finding("sys.udev", PASS, "udev rules present" if rules.exists()
                           else "no named channels required"))
    # Backends install into this repo's venv as dependency-groups (there are no per-submodule
    # venvs), so probe site-packages for a distribution only that group brings in -- the same
    # markers gui/builders.py preflights with. At most one can be present: the groups pin
    # incompatible torch/jax builds and are declared as conflicts in pyproject.toml.
    for backend, group, marker in (("openpi", "openpi", "openpi-"),
                                   ("abc", "abc-policy", "warp_lang-"),
                                   ("molmoact2", "molmoact2-train", "ai2_molmo-")):
        found = any((repo_root / ".venv").glob(f"lib/python*/site-packages/{marker}*.dist-info"))
        out.append(Finding(f"sys.backend.{backend}", PASS, "installed") if found else
                   Finding(f"sys.backend.{backend}", WARN, "not installed "
                           "(fine if you don't train/serve this backend here)",
                           f"see docs/training.md — uv sync --all-extras --group {group}"))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(config: Path, do_listen: bool) -> list[Finding]:
    findings: list[Finding] = []
    repo_root = config.resolve().parents[1] if config.parent.name == "configs" else Path.cwd()
    try:
        robot_cfg = load_station(config)
    except Exception as e:  # noqa: BLE001
        return [Finding("config", FAIL, f"cannot parse {config}: {e}",
                        "fix the YAML; see configs/station_yam.yaml comments")]
    findings.append(Finding("config", PASS, f"{config} parsed"))
    try:
        expected = expected_channels(robot_cfg)
        findings.append(Finding("config.channels", PASS,
                                f"expects: {', '.join(sorted(expected)) or '(none)'}"))
    except Exception as e:  # noqa: BLE001
        return findings + [Finding("config.channels", FAIL, f"bad robots/controllers: {e}",
                                   "check type names in configs/station_yam.yaml")]
    if robot_cfg.get("type") == "mock":
        findings.append(Finding("can", SKIP, "mock robot configured"))
    else:
        findings += check_can(expected, do_listen)
    findings += check_cameras(repo_root, robot_cfg.get("cameras_config"))
    findings += check_gpu()
    findings += check_system(repo_root, expected)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/station_yam.yaml")
    ap.add_argument("--no-listen", action="store_true",
                    help="skip the 1s-per-bus passive CAN traffic classification")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        assert classify_traffic(set()) == "silent"
        assert classify_traffic({GELLO_REPORT_ID}) == "gello"
        assert classify_traffic({0x01, 0x11}) == "motor_chain"
        assert classify_traffic({0x300}) == "other"
        print("selftest OK")
        return 0

    try:
        findings = run(Path(args.config), do_listen=not args.no_listen)
    except Exception as e:  # noqa: BLE001
        print(f"doctor crashed: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        width = max(len(f.check) for f in findings)
        icons = {PASS: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}
        for f in findings:
            line = f"[{icons[f.status]} {f.status:<4}] {f.check:<{width}}  {f.detail}"
            if f.fix:
                line += f"\n{'':{width + 10}}fix: {f.fix}"
            print(line)
        n_fail = sum(f.status == FAIL for f in findings)
        n_warn = sum(f.status == WARN for f in findings)
        print(f"\n{len(findings)} checks: {n_fail} FAIL, {n_warn} WARN")
    return 1 if any(f.status == FAIL for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
