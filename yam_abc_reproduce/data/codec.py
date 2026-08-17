"""Which H.264 encoder/decoder this machine can actually run.

Every first-party encode goes through here, so one probe decides for the whole process:
NVENC when the box has a working NVIDIA encoder, ``libx264`` otherwise.

Availability is established by *opening* the codec and pushing a frame through it -- never
by name. ``h264_nvenc`` is listed in ``av.codecs_available`` on a GPU-less box and only
fails when you try to use it, so a name check would hand half the fleet a codec it cannot
run. Decode has the opposite failure mode: ffmpeg's ``-hwaccel cuda`` prints an
initialisation error and then quietly finishes in software with exit code 0, so "it ran"
proves nothing. ``HWAccel(..., allow_software_fallback=False)`` raises instead, which is the
only way to know the GPU was really used.

PyAV bundles its own FFmpeg (7.1 at the time of writing) with libx264, NVENC and NVDEC
built in, so none of this depends on the system ffmpeg -- which on an Ubuntu 22.04 station
is 4.4.2 and cannot decode H.264 on Blackwell cards at all.

Overrides, for reproducibility and for tests that must not depend on a GPU:

    YAM_ABC_VIDEO_ENCODER   auto (default) | libx264 | h264_nvenc
    YAM_ABC_VIDEO_HWDECODE  auto (default) | 0 | 1
"""

from __future__ import annotations

import io
import os
from functools import cache, lru_cache

# Hardware first, software fallback. Order is the preference order.
ENCODERS = ("h264_nvenc", "libx264")
# NVENC refuses anything narrower than ~145px (a 64x64 probe fails on a card that encodes
# 640x480 perfectly well), so the probe frame has to clear that bar or it would reject the
# GPU on every box. Camera frames are far larger; ``open_stream`` covers the rest.
_PROBE_WH = (256, 256)

# Quality knobs per encoder. crf=18 matches what the ABC export pipeline already uses, so
# our stage costs a frame no more than theirs does. NVENC has no CRF -- vbr+cq is its
# nearest equivalent, and p4 its balanced speed/quality preset.
_QUALITY = {
    "libx264": {"crf": "18", "preset": "fast"},
    "h264_nvenc": {"rc": "vbr", "cq": "19", "preset": "p4"},
}
# Extra options for streams read back one packet at a time (the ABC release MCAP, which
# carries one CompressedVideo message per frame). bf=0 removes B-frames so packet order
# equals frame order; the repeat-headers flag puts SPS/PPS on every keyframe so any
# concatenation of packets stays decodable. The two encoders spell the latter differently.
_PER_FRAME = {
    "libx264": {"bf": "0", "x264-params": "repeat-headers=1"},
    "h264_nvenc": {"bf": "0", "repeat_spspps": "1"},
}


def encoder_options(name: str, *, per_frame_packets: bool = False) -> dict[str, str]:
    """Encoder options for ``name``; see ``_QUALITY`` / ``_PER_FRAME``."""
    opts = dict(_QUALITY.get(name, {}))
    if per_frame_packets:
        opts.update(_PER_FRAME.get(name, {}))
    return opts


def _can_encode(name: str) -> bool:
    """Can this machine really open ``name`` and push a frame through it?"""
    import av

    try:
        buf = io.BytesIO()
        with av.open(buf, mode="w", format="mp4") as container:
            stream = container.add_stream(name, rate=30, options=encoder_options(name))
            stream.width, stream.height = _PROBE_WH
            stream.pix_fmt = "yuv420p"
            for packet in stream.encode(av.VideoFrame(*_PROBE_WH, "yuv420p")):
                container.mux(packet)
            for packet in stream.encode():  # flush
                container.mux(packet)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def encoder() -> str:
    """The H.264 encoder to use, probed once per process."""
    forced = os.environ.get("YAM_ABC_VIDEO_ENCODER", "auto").strip()
    if forced and forced != "auto":
        # An explicit choice is an assertion, not a preference: someone pinning the encoder
        # for a reproducible dataset needs to hear about it rather than get a silent
        # substitute. (open_stream still degrades per-stream when a size is refused -- that
        # is a different question, and it says so when it happens.)
        if not _can_encode(forced):
            raise RuntimeError(
                f"YAM_ABC_VIDEO_ENCODER={forced} cannot encode on this machine. Unset it to "
                f"probe automatically ({' then '.join(ENCODERS)})."
            )
        print(f"[yam-abc] video encoder: {forced} (forced)", flush=True)
        return forced
    for name in ENCODERS:
        if _can_encode(name):
            print(f"[yam-abc] video encoder: {name}", flush=True)
            return name
    raise RuntimeError(
        f"no usable H.264 encoder (tried {', '.join(ENCODERS)}). PyAV ships libx264, so this "
        f"usually means a broken `av` install -- try `uv sync --all-extras`."
    )


@cache
def _opens_at(name: str, width: int, height: int) -> bool:
    """Does ``name`` accept a stream of this size? Probed on a standalone context.

    Deliberately not tried on the real output container: ``add_stream`` is lazy, so a
    rejecting encoder does not fail until the first mux -- by which point the dead stream is
    already attached and the container cannot be salvaged.
    """
    import av

    try:
        ctx = av.CodecContext.create(name, "w")
        ctx.width, ctx.height = width, height
        ctx.pix_fmt = "yuv420p"
        ctx.options = encoder_options(name)
        ctx.open()  # no close(): VideoCodecContext has none, and GC releases it
        return True
    except Exception:
        return False


def open_stream(container, *, width: int, height: int, fps: int, per_frame_packets: bool = False):
    """Add an H.264 video stream to ``container``, preferring the probed encoder.

    Falls back to ``libx264`` when the preferred encoder refuses these particular
    dimensions. ``encoder()`` proves the codec runs *at all*; it cannot prove it takes every
    frame size, and NVENC has real limits (nothing narrower than ~145px). Without this, a
    small camera would turn a working recorder into a hard failure on GPU boxes only.
    """
    name = encoder()
    if name != "libx264" and not _opens_at(name, width, height):
        print(f"[yam-abc] {name} refused {width}x{height}; using libx264", flush=True)
        name = "libx264"
    stream = container.add_stream(
        name, rate=fps, options=encoder_options(name, per_frame_packets=per_frame_packets)
    )
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    return stream


def _h264_sample() -> bytes:
    """A one-frame H.264 file, for probing the decoder."""
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = _PROBE_WH
        stream.pix_fmt = "yuv420p"
        for packet in stream.encode(av.VideoFrame(*_PROBE_WH, "yuv420p")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


@lru_cache(maxsize=1)
def _hw_decode_works() -> bool:
    import av

    setting = os.environ.get("YAM_ABC_VIDEO_HWDECODE", "auto").strip()
    if setting in ("0", "false", "no"):
        return False
    try:
        from av.codec.hwaccel import HWAccel

        with av.open(io.BytesIO(_h264_sample()), hwaccel=_cuda(HWAccel)) as container:
            decoded = any(True for _ in container.decode(video=0))
        if decoded:
            print("[yam-abc] video decode: cuda (hardware)", flush=True)
        return decoded
    except Exception:
        if setting in ("1", "true", "yes"):
            raise  # asked for hardware explicitly; do not quietly fall back to the CPU
        return False


def _cuda(hwaccel_cls):
    # allow_software_fallback=False is the whole point: without it a failed CUDA init just
    # decodes on the CPU and reports success, which is exactly how system ffmpeg misleads.
    return hwaccel_cls(device_type="cuda", allow_software_fallback=False)


def hw_decoder():
    """A fresh CUDA ``HWAccel`` when this machine really decodes H.264 on the GPU, else None.

    A new instance per container: an ``HWAccel`` carries a device context, and sharing one
    across concurrently open containers is not something PyAV promises.
    """
    if not _hw_decode_works():
        return None
    from av.codec.hwaccel import HWAccel

    return _cuda(HWAccel)
