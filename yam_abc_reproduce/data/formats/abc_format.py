"""ABC dataset format — one **MCAP** file per episode, protobuf messages.

Topics (bimanual), each carrying protobuf messages decoded via ``mcap_protobuf``:
    /instruction                       Instructions  { timestamp, data:str }
    /{left,right}-arm-state|-action    RobotState    { timestamp, repeated double position }  # 6 joints
    /{left,right}-ee-state|-action     GripperState  { timestamp, repeated double position }  # 1 = norm [0,1]
    /top-camera, /left-wrist-camera, /right-wrist-camera
                                       foxglove.CompressedVideo { timestamp, frame_id, data:h264, format:"h264" }
                                       # one message per frame
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .. import codec
from ..schema import (
    EpisodeMeta,
    action_gripper_key,
    action_joint_key,
    cam_image_key,
    gripper_pos_key,
    joint_pos_key,
)
from .base import register

_SCHEMAS = Path(__file__).with_name("abc_schemas.binpb")
# YAM-ABC-Reproduce camera role -> (ABC topic, CompressedVideo.frame_id). Note: YAM-ABC-Reproduce's
# left/right are side cameras; ABC calls its non-top cams "wrist" — we map by
# position so the topic set matches what the ABC trainer reads.
_ROLE_TOPIC = {
    "top": ("/top-camera", "top_camera-images-rgb"),
    "left": ("/left-wrist-camera", "left_camera-images-rgb"),
    "right": ("/right-wrist-camera", "right_camera-images-rgb"),
}


def _load_message_classes() -> dict:
    """Rebuild the ABC protobuf message classes from the vendored descriptor set."""
    from google.protobuf.descriptor_pb2 import FileDescriptorSet
    from google.protobuf.descriptor_pool import DescriptorPool
    from google.protobuf.message_factory import GetMessageClass

    fds = FileDescriptorSet.FromString(_SCHEMAS.read_bytes())
    pool = DescriptorPool()
    # google/protobuf/timestamp.proto may already be in the default pool; ignore dup adds.
    for fp in fds.file:
        try:
            pool.Add(fp)
        except Exception:
            pass
    return {
        name: GetMessageClass(pool.FindMessageTypeByName(name))
        for name in ("RobotState", "GripperState", "Instructions", "foxglove.CompressedVideo")
    }


def _h264_packets(frames: list, fps: int) -> list[bytes]:
    """Encode a list of HxWx3 RGB uint8 frames to an Annex-B H.264 elementary
    stream and return one packet (access unit) per frame. The encoder is whichever
    ``codec`` probed (NVENC where available); ``per_frame_packets`` supplies the bf=0 +
    repeat-headers options that keep packet order equal to frame order and put SPS/PPS on
    every keyframe, so the concatenated stream stays decodable (matches ABC's per-frame
    CompressedVideo)."""
    import av

    h, w = frames[0].shape[:2]
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="h264") as container:
        stream = codec.open_stream(
            container, width=w, height=h, fps=fps, per_frame_packets=True
        )
        packets: list[bytes] = []
        for fr in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")
            packets.extend(bytes(pkt) for pkt in stream.encode(vf))
        packets.extend(bytes(pkt) for pkt in stream.encode())  # flush lookahead
    # The reader indexes message i as frame i, so anything other than one packet per frame
    # silently misaligns video against the state/action arrays. Refuse to write that.
    if len(packets) != len(frames):
        raise RuntimeError(
            f"{stream.codec_context.name} emitted {len(packets)} packets for {len(frames)} "
            f"frames; ABC needs exactly one per frame. Force the reference encoder with "
            f"YAM_ABC_VIDEO_ENCODER=libx264 and report this."
        )
    return packets


class ABCFormat:
    def begin(self, repo_id: str, out: str | None = None) -> None:
        self._out = Path(out) if out else Path("data/abc") / repo_id
        self._out.mkdir(parents=True, exist_ok=True)
        self._ep = 0
        self._cls = _load_message_classes()

    def _stamp(self, msg, t_ns: int):
        """Set the message's google.protobuf.Timestamp field from nanoseconds."""
        msg.timestamp.seconds = t_ns // 1_000_000_000
        msg.timestamp.nanos = t_ns % 1_000_000_000
        return msg

    def add_episode(self, meta: EpisodeMeta, buffers: dict) -> None:
        from mcap_protobuf.writer import Writer as McapProtobufWriter

        RobotState = self._cls["RobotState"]
        GripperState = self._cls["GripperState"]
        Instructions = self._cls["Instructions"]
        CompressedVideo = self._cls["foxglove.CompressedVideo"]

        arms = meta.arm_names or ["left"]
        hz = max(1, int(round(meta.control_hz)))
        tick = 1_000_000_000 // hz
        n = len(buffers[joint_pos_key(arms[0])])

        path = self._out / f"episode_{self._ep:06d}.mcap"
        self._ep += 1
        with open(path, "wb") as f:
            w = McapProtobufWriter(f)
            try:
                # Task instruction (once, at t=0).
                w.write_message(
                    topic="/instruction",
                    message=self._stamp(Instructions(data=meta.task_name or ""), 0),
                    log_time=0,
                    publish_time=0,
                )
                # Per-step arm/ee state + action.
                for i in range(n):
                    t = i * tick
                    for a in arms:
                        w.write_message(
                            topic=f"/{a}-arm-state",
                            message=self._stamp(
                                RobotState(position=list(np.asarray(buffers[joint_pos_key(a)][i]).ravel())), t
                            ),
                            log_time=t, publish_time=t,
                        )
                        w.write_message(
                            topic=f"/{a}-arm-action",
                            message=self._stamp(
                                RobotState(position=list(np.asarray(buffers[action_joint_key(a)][i]).ravel())), t
                            ),
                            log_time=t, publish_time=t,
                        )
                        w.write_message(
                            topic=f"/{a}-ee-state",
                            message=self._stamp(
                                GripperState(position=list(np.asarray(buffers[gripper_pos_key(a)][i]).ravel())), t
                            ),
                            log_time=t, publish_time=t,
                        )
                        w.write_message(
                            topic=f"/{a}-ee-action",
                            message=self._stamp(
                                GripperState(position=list(np.asarray(buffers[action_gripper_key(a)][i]).ravel())), t
                            ),
                            log_time=t, publish_time=t,
                        )
                # Per-camera video, one CompressedVideo message per frame.
                for cam in meta.cameras:
                    topic, frame_id = _ROLE_TOPIC.get(
                        cam.role, (f"/{cam.role}-camera", f"{cam.role}_camera-images-rgb")
                    )
                    frames = buffers.get(cam_image_key(cam.role, "rgb"))
                    if not frames:
                        continue
                    for j, pkt in enumerate(_h264_packets(frames, hz)):
                        t = j * tick
                        w.write_message(
                            topic=topic,
                            message=self._stamp(
                                CompressedVideo(frame_id=frame_id, data=pkt, format="h264"), t
                            ),
                            log_time=t,
                            publish_time=t,
                        )
            finally:
                w.finish()

    def finalize(self) -> None:
        # One self-contained MCAP per episode; nothing to flush globally.
        pass


register("abc", ABCFormat)
