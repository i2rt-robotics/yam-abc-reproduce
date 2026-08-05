"""The V4L2 backend has to be pinned, or the mode in cameras.yaml is silently dropped.

Left to `CAP_ANY`, OpenCV opens the device with whichever backend claims it first. For a
`/dev/videoN` path -- what serial-based discovery hands us -- that is often FFMPEG or
GStreamer, and those accept every `CAP_PROP_*` set() and ignore it. The camera then streams
its own default mode: no exception, no log line, just the wrong resolution and frame rate.
That silence is the reason these are worth asserting.

`cv2.VideoCapture` is faked throughout, so none of this needs a camera (or Linux).
"""

import logging

import pytest

from yam_abc_reproduce.camera.base_v4l2 import V4L2Camera

cv2 = pytest.importorskip("cv2")


class _FakeCap:
    """Records how it was constructed and echoes back whatever was set on it."""

    def __init__(self, device, backend=None, negotiated=None):
        self.device = device
        self.backend = backend
        self.calls: list[tuple[int, float]] = []
        self.props: dict[int, float] = {}
        self._negotiated = negotiated or {}

    def set(self, prop, value):
        self.calls.append((prop, value))
        self.props[prop] = value
        return True

    def get(self, prop):
        # A real driver reports what it settled on, which need not be what was asked for.
        return self._negotiated.get(prop, self.props.get(prop, 0))

    def isOpened(self):
        return True

    def release(self):
        pass


@pytest.fixture
def capture(monkeypatch):
    """Patch cv2.VideoCapture and hand back the instance the camera built."""
    made: dict[str, _FakeCap] = {}

    def factory(negotiated=None):
        def _ctor(device, backend=None):
            made["cap"] = _FakeCap(device, backend, negotiated)
            return made["cap"]

        monkeypatch.setattr(cv2, "VideoCapture", _ctor)
        return made

    return factory


def test_capture_pins_the_v4l2_backend(capture):
    made = capture()
    V4L2Camera("left", "left", 1280, 720, 60, device="/dev/video3")
    assert made["cap"].backend == cv2.CAP_V4L2, (
        "without an explicit backend OpenCV may pick FFMPEG/GStreamer, whose "
        "CAP_PROP_* setters are silently ignored"
    )
    assert made["cap"].device == "/dev/video3"


def test_requested_mode_reaches_the_driver(capture):
    made = capture()
    V4L2Camera("left", "left", 1280, 720, 60, device=0)
    props = dict(made["cap"].calls)
    assert props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert props[cv2.CAP_PROP_FPS] == 60


def test_fourcc_is_set_before_the_mode(capture):
    """Pixel format gates which width/height/fps the driver will even offer, so MJPG has
    to land before the rest -- set afterwards, it can reset the negotiated mode."""
    made = capture()
    V4L2Camera("left", "left", 1280, 720, 60, device=0, fourcc="MJPG")
    order = [prop for prop, _ in made["cap"].calls]
    assert order.index(cv2.CAP_PROP_FOURCC) < order.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert order.index(cv2.CAP_PROP_FOURCC) < order.index(cv2.CAP_PROP_FPS)


def test_negotiated_mode_mismatch_is_reported(capture, caplog):
    """V4L2 quietly falls back to the nearest supported mode; say so."""
    made = capture(
        {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,
        }
    )
    with caplog.at_level(logging.WARNING):
        V4L2Camera("left", "left", 1280, 720, 60, device="/dev/video3")
    assert made["cap"].backend == cv2.CAP_V4L2
    assert "asked for 1280x720@60fps" in caplog.text
    assert "640x480@30fps" in caplog.text


def test_no_warning_when_the_driver_honours_the_request(capture, caplog):
    capture()  # echoes back what was set, i.e. an exact match
    with caplog.at_level(logging.WARNING):
        V4L2Camera("left", "left", 1280, 720, 60, device=0)
    assert caplog.text == ""


def test_unreadable_fps_is_not_treated_as_a_mismatch(capture, caplog):
    """Some UVC drivers report fps 0 even when the mode took. That must not warn on its
    own, or every camera on such a driver cries wolf on every start."""
    made = capture({cv2.CAP_PROP_FPS: 0})
    with caplog.at_level(logging.WARNING):
        V4L2Camera("left", "left", 1280, 720, 60, device=0)
    assert made["cap"].props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert caplog.text == ""
