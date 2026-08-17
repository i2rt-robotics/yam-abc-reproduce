"""The GUI's policy-server liveness poll must not make the server log a traceback.

A connection that hangs up before sending an HTTP request line makes a websockets server
log a full handshake-failure traceback -- and _port_probe polls every few seconds, which
buried the real log badly enough to look like a crash. Probed against our own _wire
server, which is what the abc/molmoact2 backends serve with.
"""

import logging
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("openpi_client")
from fastapi.testclient import TestClient  # noqa: E402

from yam_abc_reproduce.config import CameraConfig, RobotConfig, StationConfig  # noqa: E402
from yam_abc_reproduce.deploy.servers import _wire  # noqa: E402
from yam_abc_reproduce.gui.server import create_app  # noqa: E402

_PORT = 8399  # not in _DEPLOY_PORT, so a real server on this box cannot collide


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def wire_server():
    t = threading.Thread(
        target=_wire.serve,
        args=(lambda obs: {"actions": [[0.0]]}, "127.0.0.1", _PORT),
        kwargs={"metadata": {"backend": "test"}},
        daemon=True,
    )
    t.start()
    time.sleep(0.6)  # let it bind
    yield


@pytest.fixture
def client(tmp_path):
    cfg = StationConfig(
        robot=RobotConfig(type="mock", num_arm_joints=6),
        cameras=[CameraConfig(name="top", type="mock", role="top", mode="mono")],
        control_hz=200.0,
        save_root=str(tmp_path / "episodes"),
    )
    with TestClient(create_app(cfg, mock=True)) as c:
        yield c


def test_liveness_poll_logs_no_traceback(client, wire_server):
    ws_log = logging.getLogger("websockets")
    handler = _Collect()
    ws_log.addHandler(handler)
    prev = ws_log.level
    ws_log.setLevel(logging.INFO)
    try:
        r = client.get(f"/api/deploy/server-status?host=127.0.0.1&port={_PORT}")
        time.sleep(0.5)  # the server logs from its own thread
    finally:
        ws_log.removeHandler(handler)
        ws_log.setLevel(prev)

    assert r.status_code == 200
    assert r.json()["listening"] is True, "the probe must still detect the server"
    tracebacks = [rec for rec in handler.records if rec.exc_info]
    assert not tracebacks, f"probe provoked {len(tracebacks)}: {tracebacks[0].getMessage()}"
