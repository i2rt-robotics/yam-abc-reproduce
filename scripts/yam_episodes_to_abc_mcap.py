#!/usr/bin/env python3
"""Convert raw YAM bimanual teleop episode directories into ABC release-format
MCAP episodes (the input that ``third_party/policy/abc/export_mcap.py`` expects).

This is a data-prep utility for feeding externally-collected YAM demonstrations
into the ABC training pipeline. It exists because those recordings store one MCAP
per arm plus separate per-camera video, whereas ABC's ``export_mcap.py`` reads a
single ``episode.mcap`` carrying every state/action/camera stream on the release
topic names.

Source episode directory (``episode_*.npy.mp4/``):
    left.mcap    /left-robot-state           (RobotState, 6 arm joints)   -> OBS state
                 /left-gripper-state          (GripperState, 1 in [0,1])  -> OBS state
                 /left-command-state          (RobotState, 6 abs joints)  -> ACTION
                 /left-command-gripper-state  (GripperState, 1)           -> ACTION
    right.mcap   symmetric
    camera_{top,left,right}-images-rgb.mp4    (h264) + -timestamp.npy (epoch seconds)

    NOTE on the action source: ``/*-command-state`` is the absolute joint command
    sent to the follower and lives in the same space as the observed state -- this
    matches ABC's ``/*-arm-action``. The separate ``/action-*-robot-state`` stream
    is an end-effector twist/delta representation (verified: it leaves the absolute
    joint manifold), so it is NOT used here.

Target (one ``episode.mcap`` per episode):
    /instruction                         Instructions{data:str}
    /{left,right}-arm-state|-action      RobotState{position:6}
    /{left,right}-ee-state|-action       GripperState{position:1 in [0,1]}
    /top-camera,/left-wrist-camera,/right-wrist-camera
                                         foxglove.CompressedVideo{data:h264,format,frame_id}

Output layout (consumed by export_mcap.py, one root per split):
    <out>/{train,val}/<task>/episode_NNNNNN/episode.mcap

Then build the ABC training cache (needs `uv sync --group abc-policy`; run from the abc
submodule so its abc_minimal/ package is importable, using this repo's venv):
    cd third_party/policy/abc
    ../../../.venv/bin/python export_mcap.py <out>/train  <ABC_CACHE>/train_real 4
    ../../../.venv/bin/python export_mcap.py <out>/val    <ABC_CACHE>/val_real   2
The trainer derives each episode's prompt from its directory <task> name
(underscores -> spaces), so name <task> exactly as the desired instruction.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

_SCHEMAS = Path(__file__).resolve().parents[1] / "yam_abc_reproduce" / "data" / "formats" / "abc_schemas.binpb"

# release topic  <-  (source mcap file, {kind: source topic})
_STATE_MAP = {
    "left":  ("left.mcap",  {"arm-state": "/left-robot-state", "arm-action": "/left-command-state",
                             "ee-state": "/left-gripper-state", "ee-action": "/left-command-gripper-state"}),
    "right": ("right.mcap", {"arm-state": "/right-robot-state", "arm-action": "/right-command-state",
                             "ee-state": "/right-gripper-state", "ee-action": "/right-command-gripper-state"}),
}
# source camera key -> (release topic, CompressedVideo.frame_id)
_CAM_MAP = {
    "camera_top":   ("/top-camera", "top_camera-images-rgb"),
    "camera_left":  ("/left-wrist-camera", "left_camera-images-rgb"),
    "camera_right": ("/right-wrist-camera", "right_camera-images-rgb"),
}
_REQUIRED = {
    "left.mcap":  ["/left-robot-state", "/left-command-state", "/left-gripper-state", "/left-command-gripper-state"],
    "right.mcap": ["/right-robot-state", "/right-command-state", "/right-gripper-state", "/right-command-gripper-state"],
}


def _load_classes() -> dict:
    from google.protobuf.descriptor_pb2 import FileDescriptorSet
    from google.protobuf.descriptor_pool import DescriptorPool
    from google.protobuf.message_factory import GetMessageClass
    fds = FileDescriptorSet.FromString(_SCHEMAS.read_bytes())
    pool = DescriptorPool()
    for fp in fds.file:
        try:
            pool.Add(fp)
        except Exception:
            pass  # e.g. google/protobuf/timestamp.proto may already be registered
    return {n: GetMessageClass(pool.FindMessageTypeByName(n))
            for n in ("RobotState", "GripperState", "Instructions", "foxglove.CompressedVideo")}


def _read_topic(mcap_path: Path, topic: str):
    """Return (log_times_ns[int64], positions[float64 (N,D)]) sorted by time."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    ts, pos = [], []
    with open(mcap_path, "rb") as f:
        for _, ch, m, dec in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            if ch.topic == topic:
                ts.append(m.log_time)
                pos.append(list(dec.position))
    if not ts:
        return np.empty(0, np.int64), np.empty((0, 0))
    order = np.argsort(ts)
    return np.array(ts, np.int64)[order], np.array(pos, np.float64)[order]


def _mp4_to_h264_packets(mp4_path: Path, fps: int = 30) -> list[bytes]:
    """Decode a source mp4 and re-encode to an Annex-B H.264 elementary stream,
    one packet (access unit) per frame. The encoder is whichever ``codec`` probed (NVENC
    where available); ``per_frame_packets`` supplies bf=0 + repeat-headers, so packet order
    == frame order and SPS/PPS repeat on every keyframe (the convention export_mcap.py
    assumes when it re-reads the frames). Frames are streamed rather than collected, so an
    episode never has to fit in memory twice."""
    import av

    from yam_abc_reproduce.data import codec

    with av.open(str(mp4_path)) as probe:
        hw = codec.hw_decoder() if probe.streams.video[0].codec_context.name == "h264" else None

    buf = io.BytesIO()
    packets: list[bytes] = []
    n_frames = 0
    with av.open(str(mp4_path), hwaccel=hw) as inc, av.open(buf, mode="w", format="h264") as out:
        ost = None
        for frame in inc.decode(inc.streams.video[0]):
            arr = frame.to_ndarray(format="rgb24")
            if ost is None:
                h, w = arr.shape[:2]
                ost = codec.open_stream(
                    out, width=w, height=h, fps=fps, per_frame_packets=True
                )
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            packets.extend(bytes(pkt) for pkt in ost.encode(vf))
            n_frames += 1
        if ost is not None:
            packets.extend(bytes(pkt) for pkt in ost.encode())  # flush lookahead
    # export_mcap.py indexes message i as frame i; anything else silently misaligns video
    # against the state/action arrays.
    if len(packets) != n_frames:
        raise RuntimeError(
            f"encoder emitted {len(packets)} packets for {n_frames} frames in {mp4_path}; "
            f"ABC needs exactly one per frame. Retry with YAM_ABC_VIDEO_ENCODER=libx264."
        )
    return packets


def _is_valid(ep: Path) -> bool:
    if not (ep / "write_complete.flag").exists() or (ep / "discarded.flag").exists():
        return False
    if any(not (ep / fn).exists() for fn in _REQUIRED):
        return False
    return all((ep / f"{c}-images-rgb.mp4").exists() and (ep / f"{c}-timestamp.npy").exists() for c in _CAM_MAP)


def _find_episodes(src: Path) -> list[Path]:
    """Accept a flat dir of ``episode_*.npy.mp4`` dirs, else one/two levels nested."""
    for pat in ("episode_*.npy.mp4", "*/episode_*.npy.mp4", "*/*/episode_*.npy.mp4"):
        eps = [e for e in sorted(src.glob(pat)) if e.is_dir()]
        if eps:
            return [e for e in eps if _is_valid(e)]
    return []


def _stamp(msg, t_ns: int):
    msg.timestamp.seconds = int(t_ns) // 1_000_000_000
    msg.timestamp.nanos = int(t_ns) % 1_000_000_000
    return msg


def convert_episode(ep: Path, out_mcap: Path, task: str, cls: dict) -> None:
    from mcap_protobuf.writer import Writer
    RobotState, GripperState = cls["RobotState"], cls["GripperState"]
    Instructions, CompressedVideo = cls["Instructions"], cls["foxglove.CompressedVideo"]

    out_mcap.parent.mkdir(parents=True, exist_ok=True)
    t0_hint = None
    with open(out_mcap, "wb") as f:
        w = Writer(f)
        try:
            for side, (fn, topics) in _STATE_MAP.items():
                p = ep / fn
                for kind, src_topic in topics.items():
                    ts, pos = _read_topic(p, src_topic)
                    if len(ts) == 0:
                        raise RuntimeError(f"empty {src_topic}")
                    is_ee = kind.startswith("ee")
                    Msg = GripperState if is_ee else RobotState
                    if is_ee:
                        pos = np.clip(pos, 0.0, 1.0)
                    if t0_hint is None or ts[0] < t0_hint:
                        t0_hint = int(ts[0])
                    topic = f"/{side}-{kind}"
                    for t, row in zip(ts, pos):
                        w.write_message(topic=topic, message=_stamp(Msg(position=list(row)), int(t)),
                                        log_time=int(t), publish_time=int(t))
            w.write_message(topic="/instruction",
                            message=_stamp(Instructions(data=task.replace("_", " ")), t0_hint or 0),
                            log_time=int(t0_hint or 0), publish_time=int(t0_hint or 0))
            for cam, (topic, frame_id) in _CAM_MAP.items():
                cam_ts = np.load(ep / f"{cam}-timestamp.npy").astype(np.float64)
                packets = _mp4_to_h264_packets(ep / f"{cam}-images-rgb.mp4")
                if not packets:
                    raise RuntimeError(f"no frames from {cam}")
                if len(packets) == len(cam_ts):
                    log_ns = (cam_ts * 1e9).astype(np.int64)
                else:  # packet/frame drift: respace evenly over the real capture window
                    log_ns = (np.linspace(cam_ts[0], cam_ts[-1], len(packets)) * 1e9).astype(np.int64)
                for pkt, t in zip(packets, log_ns):
                    w.write_message(
                        topic=topic,
                        message=_stamp(CompressedVideo(frame_id=frame_id, data=pkt, format="h264"), int(t)),
                        log_time=int(t), publish_time=int(t))
        finally:
            w.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw YAM episodes -> ABC release-format episode.mcap")
    ap.add_argument("--src", required=True, help="dir containing episode_*.npy.mp4 episode dirs")
    ap.add_argument("--out", required=True, help="release-mcap root (writes <out>/{train,val}/<task>/...)")
    ap.add_argument("--task", default="insert_the_wireless_bluetooth_earbuds_into_the_charging_case",
                    help="task/instruction; also the dir name the trainer reads for the prompt")
    ap.add_argument("--val", type=int, default=2, help="episodes assigned to the val split")
    ap.add_argument("--limit", type=int, default=0, help="cap number of episodes (0 = all)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    eps = _find_episodes(src)
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        print(f"no valid episodes under {src}")
        return
    print(f"{len(eps)} episodes -> {out}  (val={min(args.val, max(0, len(eps)-1))}, task={args.task})")
    cls = _load_classes()
    n_val = min(args.val, max(0, len(eps) - 1))
    ok = 0
    for k, ep in enumerate(eps):
        split = "val" if k < n_val else "train"
        dst = out / split / args.task / f"episode_{k:06d}" / "episode.mcap"
        try:
            convert_episode(ep, dst, args.task, cls)
            ok += 1
            print(f"[OK] {split}/episode_{k:06d}  <- {ep.name}")
        except Exception as ex:
            print(f"[FAIL] {ep.name}: {ex}")
    print(f"\nwrote {ok}/{len(eps)} episodes")
    print(f"  train: {out}/train/{args.task}/")
    print(f"  val:   {out}/val/{args.task}/")


if __name__ == "__main__":
    main()
