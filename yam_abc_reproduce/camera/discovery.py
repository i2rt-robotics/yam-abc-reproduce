"""Enumerate connected cameras (RealSense + V4L2/UVC decxin) with serials.

Shared by the ``yam_abc_reproduce cameras`` CLI (prints the list) and the GUI's
``/api/cameras/detect`` endpoint (populates the Station rail's serial pickers),
so a station is configured by picking from live hardware instead of hand-typing
serials into ``cameras.yaml``.
"""

from __future__ import annotations


def discover_realsense() -> list[dict]:
    """RealSense devices via pyrealsense2. Empty list if the lib/hardware is absent."""
    try:
        import pyrealsense2 as rs
    except Exception:
        return []
    out: list[dict] = []
    try:
        for d in rs.context().query_devices():
            out.append(
                {
                    "kind": "realsense",
                    "serial": d.get_info(rs.camera_info.serial_number),
                    "product": d.get_info(rs.camera_info.name),
                    "node": None,
                    # RealSense is the only mono type we can positively identify here.
                    "suggested_type": "realsense",
                }
            )
    except Exception:
        pass
    return out


def discover_v4l2() -> list[dict]:
    """V4L2/UVC devices (decxin, and RealSense's UVC nodes) via pyudev, deduped by
    serial to the lowest-numbered /dev/videoN (one camera exposes several nodes)."""
    try:
        import pyudev
    except Exception:
        return []
    by_serial: dict[str, dict] = {}
    for dev in pyudev.Context().list_devices(subsystem="video4linux"):
        node = dev.device_node
        if not node:
            continue
        serial = dev.get("ID_SERIAL_SHORT") or dev.get("ID_SERIAL")
        if not serial:
            continue
        product = dev.get("ID_V4L_PRODUCT") or dev.get("ID_MODEL") or "?"
        prev = by_serial.get(serial)
        if prev is None or node < prev["node"]:
            by_serial[serial] = {
                "kind": "v4l2",
                "serial": serial,
                "product": product,
                "node": node,
                # A UVC device could be mono or stereo; default to the common case.
                "suggested_type": "decxin_mono",
            }
    return sorted(by_serial.values(), key=lambda d: d["node"])


def discover_cameras() -> list[dict]:
    """All connected cameras, RealSense first then V4L2. Each entry:
    ``{kind, serial, product, node, suggested_type}``."""
    return discover_realsense() + discover_v4l2()
