"""Minimal MP4 read/write helpers for the canonical (default) on-disk format.

Uses PyAV. Video is written as H.264 with whichever encoder ``codec`` probed -- NVENC on a
box that has one, ``libx264`` otherwise. OpenCV cannot do this job: the opencv-python wheel's
bundled FFmpeg has no ``libx264``, so every H.264 fourcc falls through to ``h264_v4l2m2m``
and fails to open a writer at all.

Reading handles H.264 *and* the MPEG-4 Part 2 that episodes recorded before this change
carry, so nothing already on disk becomes unreadable.

The LeRobot path does NOT go through here -- LeRobot encodes its own video from raw frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import codec


def write_mp4(path: str | Path, frames: list[np.ndarray], fps: int) -> None:
    """Write a list of HxWx3 RGB uint8 frames to an H.264 MP4."""
    import av

    if not frames:
        raise ValueError(f"no frames to write for {path}")
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = codec.open_stream(container, width=w, height=h, fps=int(fps))
        for fr in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder's lookahead
            container.mux(packet)


def video_codec_name(path: str | Path) -> str:
    """Codec of the first video stream (``"h264"``, ``"mpeg4"``, ...), or "" if there is none."""
    import av

    with av.open(str(path)) as container:
        streams = container.streams.video
        return streams[0].codec_context.name if streams else ""


def transcode_to_h264(src: str | Path, dst: str | Path) -> None:
    """Re-encode a video file to H.264, for browsers that cannot play what it holds.

    Only episodes recorded before the recorder emitted H.264 itself need this -- a ``<video>``
    tag will not play MPEG-4 Part 2. ``+faststart`` moves the moov atom to the front so
    playback can start before the whole file has arrived.
    """
    import av

    with av.open(str(src)) as inc, av.open(
        str(dst), mode="w", options={"movflags": "+faststart"}
    ) as out:
        istream = inc.streams.video[0]
        fps = int(round(float(istream.average_rate or 30)))
        ostream = None
        for frame in inc.decode(istream):
            if ostream is None:
                ostream = codec.open_stream(
                    out, width=frame.width, height=frame.height, fps=fps
                )
            # Keep the decoded frame's own pts: PyAV rescales it from the frame's time_base
            # to the stream's. Clearing it and letting the encoder assign its own loses a
            # frame under NVENC -- 8 packets go in, 7 come back out.
            for packet in ostream.encode(frame):
                out.mux(packet)
        if ostream is not None:
            for packet in ostream.encode():
                out.mux(packet)


def _decoded_frames(container) -> list[np.ndarray]:
    return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def read_mp4(path: str | Path) -> list[np.ndarray]:
    """Decode an MP4 back to a list of HxWx3 RGB uint8 frames.

    Hardware decode is used only for H.264. Older episodes are MPEG-4 Part 2, which has no
    reliable CUDA path, and the ``HWAccel`` is deliberately built with
    ``allow_software_fallback=False`` -- so handing it one would fail the read outright
    rather than quietly decode on the CPU.
    """
    import av

    path = str(path)
    with av.open(path) as container:
        streams = container.streams.video
        hw = codec.hw_decoder() if streams and streams[0].codec_context.name == "h264" else None
        if hw is None:
            return _decoded_frames(container)
    # Hardware path needs the accelerator supplied at open time, so re-open with it. Only
    # the header is parsed twice; the frames are decoded once.
    with av.open(path, hwaccel=hw) as container:
        return _decoded_frames(container)
