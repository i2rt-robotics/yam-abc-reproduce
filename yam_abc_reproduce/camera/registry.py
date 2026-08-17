"""Build camera drivers from CameraConfig entries."""

from __future__ import annotations

from ..config import CameraConfig
from .interface import CameraDriver


def build_camera(cfg: CameraConfig) -> CameraDriver:
    if cfg.type == "decxin_mono":
        from .decxin_mono import DecxinMono

        return DecxinMono(cfg.name, cfg.role, cfg.width, cfg.height, cfg.fps, serial=cfg.serial)
    if cfg.type == "decxin_stereo":
        from .decxin_stereo import DecxinStereo

        return DecxinStereo(cfg.name, cfg.role, cfg.width, cfg.height, cfg.fps, serial=cfg.serial)
    if cfg.type == "realsense":
        from .realsense import RealSense

        return RealSense(
            cfg.name,
            cfg.role,
            cfg.width,
            cfg.height,
            cfg.fps,
            serial=cfg.serial,
            enable_depth=cfg.enable_depth,
        )
    if cfg.type == "mock":
        from .interface import CameraMode
        from .mock_camera import MockCamera

        return MockCamera(cfg.name, cfg.role, CameraMode(cfg.mode), cfg.width, cfg.height)
    raise ValueError(f"unknown camera type {cfg.type!r}")


def build_cameras(configs: list[CameraConfig]) -> list[CameraDriver]:
    """Open all cameras, cleaning up already-opened ones if any fail. Errors are
    re-raised with the offending camera's name/serial for a clear message."""
    built: list[CameraDriver] = []
    for cfg in configs:
        try:
            built.append(build_camera(cfg))
        except Exception as exc:
            for d in built:
                try:
                    d.stop()
                except Exception:
                    pass
            raise RuntimeError(
                f"failed to open camera {cfg.name!r} "
                f"(type={cfg.type}, serial={cfg.serial}): {exc}"
            ) from exc
    return built
