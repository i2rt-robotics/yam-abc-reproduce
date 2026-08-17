"""Codec selection and the H.264 recorder format.

Everything here has to pass on a box with no GPU (CI, dev machines), so nothing may *require*
NVENC -- the hardware paths are exercised by faking the probe result, and the real hardware
check lives in the PR's manual verification.
"""

import numpy as np
import pytest

pytest.importorskip("av")

import av  # noqa: E402

from yam_abc_reproduce.data import codec, video  # noqa: E402


def _frames(n=12, w=320, h=240):
    """Frames with structure, so a lossy round-trip can still be compared meaningfully."""
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 0] = (i * 20) % 256      # flat, survives compression
        f[: h // 2, :, 1] = 200          # a hard edge
        out.append(f)
    return out


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    """The probes are process-cached; clear them so each test picks up its own env."""
    # Bind the real cached functions now: a test may monkeypatch the module attribute, and
    # teardown must still clear the original rather than whatever replaced it.
    cached = (codec.encoder, codec._hw_decode_works, codec._opens_at)
    monkeypatch.delenv("YAM_ABC_VIDEO_ENCODER", raising=False)
    monkeypatch.delenv("YAM_ABC_VIDEO_HWDECODE", raising=False)
    for fn in cached:
        fn.cache_clear()
    yield
    for fn in cached:
        fn.cache_clear()


def test_probe_falls_back_to_software_when_nvenc_cannot_open(monkeypatch):
    """h264_nvenc is listed in av.codecs_available on GPU-less boxes and only fails when
    used, so selection has to come from opening it -- not from the name list."""
    monkeypatch.setattr(codec, "_can_encode", lambda name: name == "libx264")
    assert codec.encoder() == "libx264"


def test_probe_prefers_hardware_when_it_works(monkeypatch):
    monkeypatch.setattr(codec, "_can_encode", lambda name: True)
    assert codec.encoder() == "h264_nvenc"  # first entry of ENCODERS


def test_encoder_override_is_honoured(monkeypatch):
    monkeypatch.setenv("YAM_ABC_VIDEO_ENCODER", "libx264")
    monkeypatch.setattr(codec, "_can_encode", lambda name: True)  # hardware would win otherwise
    assert codec.encoder() == "libx264"


def test_forcing_an_unusable_encoder_raises(monkeypatch):
    """Pinning the encoder is an assertion, not a preference: someone fixing it for a
    reproducible dataset must not silently get a substitute."""
    monkeypatch.setenv("YAM_ABC_VIDEO_ENCODER", "h264_nvenc")
    monkeypatch.setattr(codec, "_can_encode", lambda name: False)
    with pytest.raises(RuntimeError, match="cannot encode on this machine"):
        codec.encoder()


def test_hwdecode_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("YAM_ABC_VIDEO_HWDECODE", "0")
    assert codec.hw_decoder() is None


def test_per_frame_options_carry_the_repeat_header_flag():
    """ABC needs SPS/PPS on every keyframe and no B-frames; the two encoders spell the
    former differently, which is why callers must not hardcode either."""
    x264 = codec.encoder_options("libx264", per_frame_packets=True)
    nvenc = codec.encoder_options("h264_nvenc", per_frame_packets=True)
    assert x264["bf"] == "0" and "repeat-headers=1" in x264["x264-params"]
    assert nvenc["bf"] == "0" and nvenc["repeat_spspps"] == "1"
    # and plain file writes must not inherit them
    assert "bf" not in codec.encoder_options("libx264")


def test_open_stream_falls_back_when_the_encoder_refuses_the_size(monkeypatch, capsys):
    """NVENC rejects anything narrower than ~145px. The probe proves the codec runs at all,
    not that it takes every frame size, so the writer has to degrade rather than die."""
    monkeypatch.setattr(codec, "encoder", lambda: "h264_nvenc")
    monkeypatch.setattr(codec, "_opens_at", lambda name, w, h: False)
    with av.open(__import__("io").BytesIO(), mode="w", format="mp4") as container:
        stream = codec.open_stream(container, width=64, height=64, fps=30)
        assert stream.codec_context.name == "libx264"
    assert "using libx264" in capsys.readouterr().out


def test_recorder_writes_h264(tmp_path):
    """The whole point: episodes stop being MPEG-4 Part 2. OpenCV could not do this at all --
    its bundled FFmpeg has no libx264, so every H.264 fourcc fails to open a writer."""
    p = tmp_path / "ep.mp4"
    video.write_mp4(p, _frames(), fps=30)
    assert video.video_codec_name(p) == "h264"


def test_round_trip_keeps_every_frame(tmp_path):
    frames = _frames(n=20)
    p = tmp_path / "ep.mp4"
    video.write_mp4(p, frames, fps=30)
    back = video.read_mp4(p)
    assert len(back) == len(frames)
    assert back[0].shape == frames[0].shape and back[0].dtype == np.uint8
    # H.264 is lossy, so compare with tolerance rather than exactly.
    drift = max(float(np.abs(a.astype(int) - b.astype(int)).mean()) for a, b in zip(frames, back))
    assert drift < 6, f"mean per-frame drift {drift} is larger than compression should cause"


def test_legacy_mp4v_episodes_still_decode(tmp_path):
    """Every episode recorded before this change is MPEG-4 Part 2. If read_mp4 stopped
    handling it, all existing data would silently become unconvertible."""
    frames = _frames(n=8)
    p = tmp_path / "legacy.mp4"
    with av.open(str(p), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    assert video.video_codec_name(p) == "mpeg4"
    assert len(video.read_mp4(p)) == len(frames)


def test_transcode_produces_playable_h264(tmp_path):
    src = tmp_path / "legacy.mp4"
    video.write_mp4(src, _frames(n=8), fps=30)  # content does not matter, only the re-encode
    dst = tmp_path / "out.mp4"
    video.transcode_to_h264(src, dst)
    assert video.video_codec_name(dst) == "h264"
    assert len(video.read_mp4(dst)) == 8


def test_abc_emits_exactly_one_packet_per_frame():
    """abc_format's reader indexes CompressedVideo message i as frame i. Any other packet
    count silently misaligns the video against the state/action arrays."""
    from yam_abc_reproduce.data.formats.abc_format import _h264_packets

    frames = _frames(n=15)
    packets = _h264_packets(frames, fps=30)
    assert len(packets) == len(frames)
    # Annex-B start code, and SPS first so a concatenation of packets stays decodable.
    assert packets[0][:4] == b"\x00\x00\x00\x01"
    assert packets[0][4] & 0x1F == 7
