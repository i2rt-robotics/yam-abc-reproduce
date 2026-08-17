"""M5: GUI backend smoke test against a mock session (no hardware)."""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from yam_abc_reproduce.config import CameraConfig, RobotConfig, StationConfig  # noqa: E402
from yam_abc_reproduce.gui.server import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    cfg = StationConfig(
        robot=RobotConfig(type="mock", num_arm_joints=6),
        cameras=[CameraConfig(name="top", type="mock", role="top", mode="mono")],
        control_hz=200.0,
        save_root=str(tmp_path / "episodes"),
    )
    app = create_app(cfg, mock=True)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["mock"] is True


def test_teleop_record_and_preview(client):
    assert client.post("/api/collect/start-teleop").json()["teleop_running"] is True
    time.sleep(0.2)  # let the loop push some frames

    # Live JPEG preview is served.
    img = client.get("/api/cameras/top/preview.jpg")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/jpeg"
    assert img.content[:2] == b"\xff\xd8"  # JPEG magic

    # Record a short episode.
    assert "episode" in client.post(
        "/api/collect/start-recording", json={"task_name": "pp"}
    ).json()
    time.sleep(0.2)
    res = client.post("/api/collect/stop-recording").json()
    assert res["frames"] > 0

    client.post("/api/collect/stop-teleop")
    assert client.get("/api/collect/episodes").json()["count"] == 1


def test_ws_status(client):
    with client.websocket_connect("/ws") as ws:
        s = ws.receive_json()
        assert "teleop_running" in s and "cameras" in s


def test_job_lifecycle(client):
    job = client.post("/api/jobs", json={"kind": "train", "params": {"steps": 3}}).json()
    if "is not installed" in (job.get("detail") or ""):
        pytest.skip(job["detail"].split(".")[0])  # no policy backend synced into this venv
    jid = job["id"]
    # Wait for the stub job to emit logs and exit.
    deadline = time.time() + 5
    logs = []
    while time.time() < deadline:
        logs = client.get(f"/api/jobs/{jid}/logs").json()["lines"]
        if client.get(f"/api/jobs/{jid}").json()["status"] == "exited":
            break
        time.sleep(0.2)
    assert any("step" in line for line in logs)
    assert client.post(f"/api/jobs/{jid}/stop").json()["stopped"] is True


def test_deploy_session_refuses_teleop_over_http(client):
    """A rollout runs on the follower buses only, so Start Teleop can't just re-enable
    sync — it needs a rebuild. The route must say so with a 409, not a 500."""
    app = client.app
    app.state.session.go_live(app.state.session.cfg, followers_only=True)

    r = client.post("/api/collect/start-teleop")
    assert r.status_code == 409
    assert "Reset Session" in r.json()["detail"]

    status = client.get("/api/collect/status").json()
    assert status["followers_only"] is True and status["live"] is True

    # Reset Session is the documented way out, and it restores teleop.
    assert client.post("/api/session/reset").json()["ok"] is True
    assert client.post("/api/collect/start-teleop").status_code == 200
    assert client.get("/api/collect/status").json()["followers_only"] is False


def test_config_exposes_the_station_home_pose(tmp_path):
    """The Deploy tab pre-fills its home-pose field from the station config."""
    from yam_abc_reproduce.config import RobotConfig, StationConfig
    from yam_abc_reproduce.gui.server import create_app

    pose = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0] * 2
    cfg = StationConfig(
        robot=RobotConfig(type="mock", num_arm_joints=6),
        save_root=str(tmp_path / "episodes"),
        deploy_home_pose=pose,
    )
    with TestClient(create_app(cfg, mock=True)) as c:
        assert c.get("/api/config").json()["deploy_home_pose"] == pose


def test_openpi_train_command_makes_gcsfs_proxy_aware():
    """The openpi train command must export FSSPEC_GS as a *shell-quoted JSON dict*.

    Two ways to break gcsfs's proxy handling silently: shell-mangled quoting, or the
    FSSPEC_GS_SESSION_KWARGS spelling, which fsspec passes through as a raw string.
    """
    import json
    import subprocess

    from yam_abc_reproduce.gui import builders

    cmd, script = builders._train_openpi({"dataset": ""}, "pi0_yam")
    assert "FSSPEC_GS_SESSION_KWARGS" not in script
    # Let the real shell do the parsing, then read back what the child would see.
    prefix = script.split(" && echo ")[0]
    out = subprocess.run(
        [*cmd[:2], f'{prefix} && printf %s "$FSSPEC_GS"'],
        capture_output=True, text=True, timeout=30,
    ).stdout
    assert json.loads(out) == {"session_kwargs": {"trust_env": True}}


# --- GPU pinning (issue #3) --------------------------------------------------
# The GPUs field used to only *size* a run: openpi built its mesh as
# (jax.device_count() // fsdp_devices, fsdp_devices), so "GPUs: 1" on an 8-card box was still
# 8-way data parallelism across every card. These lock in that the launch commands confine
# themselves. nvidia-smi is absent on plenty of dev boxes, so every case fakes the inventory.

def _fake_nvidia_smi(stdout):
    """Stand in for subprocess.run so the parser can be driven with real nvidia-smi output."""
    return lambda *a, **kw: type("R", (), {"stdout": stdout})()


@pytest.fixture
def no_gpus(monkeypatch):
    """No nvidia-smi and no inherited CUDA_VISIBLE_DEVICES: the deterministic fallback."""
    from yam_abc_reproduce.gui import gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpus, "inventory", list)
    return gpus


def test_train_commands_pin_cuda_visible_devices(no_gpus):
    from yam_abc_reproduce.gui import builders

    for _, script in (builders._train_openpi({"dataset": "", "gpus": "2"}, "pi0_yam"),
                      builders._train_molmoact2({"gpus": "2"}),
                      builders._train_abc({"gpus": "2"})):
        assert "export CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID " in script


def test_openpi_pins_the_gpus_before_the_norm_stats_pass(no_gpus):
    """compute_norm_stats.py initialises JAX too, so the pin has to be in the shared export
    rather than on the train.py line -- otherwise the norm-stats pass opens every card."""
    import subprocess

    from yam_abc_reproduce.gui import builders

    cmd, script = builders._train_openpi({"dataset": ""}, "pi0_yam")
    assert script.index("CUDA_VISIBLE_DEVICES") < script.index("compute_norm_stats.py")
    # Let the real shell parse the export, then read back what both children would see.
    prefix = script.split(" && echo ")[0]
    out = subprocess.run(
        [*cmd[:2], f'{prefix} && printf %s "$CUDA_VISIBLE_DEVICES,$CUDA_DEVICE_ORDER"'],
        capture_output=True, text=True, timeout=30,
    ).stdout
    assert out == "0,PCI_BUS_ID"


def test_train_picks_the_freest_gpus(monkeypatch):
    from yam_abc_reproduce.gui import builders, gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # nvidia-smi enumeration order, ascending index; card 0 is busy.
    monkeypatch.setattr(gpus, "inventory", lambda: [("0", 500), ("1", 31000), ("5", 32000)])
    assert "CUDA_VISIBLE_DEVICES=5 " in builders._train_abc({"gpus": "1"})[1]
    # Two freest, rendered in enumeration order rather than free-VRAM order.
    assert "CUDA_VISIBLE_DEVICES=1,5 " in builders._train_abc({"gpus": "2"})[1]


def test_train_subsets_an_inherited_cuda_visible_devices(monkeypatch):
    """README/docs tell operators to export CUDA_VISIBLE_DEVICES *before* starting the GUI to
    choose which cards. That list is a ceiling -- never replaced with 0..N-1."""
    from yam_abc_reproduce.gui import builders, gpus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3, 5,7")  # a stray space is legal
    monkeypatch.setattr(gpus, "inventory", list)          # no nvidia-smi to rank with
    assert "CUDA_VISIBLE_DEVICES=3,5 " in builders._train_abc({"gpus": "2"})[1]
    # Ranking with nvidia-smi never reaches outside the offered set either.
    monkeypatch.setattr(gpus, "inventory", lambda: [("3", 100), ("5", 200), ("7", 32000)])
    assert "CUDA_VISIBLE_DEVICES=7 " in builders._train_abc({"gpus": "1"})[1]


def test_train_gpus_defaults_to_one(no_gpus):
    """A cleared field arrives as "" and falls back in the builder. That fallback used to be
    8 for molmoact2/abc, which would now silently claim the whole box."""
    from yam_abc_reproduce.gui import builders

    for build in (builders._train_molmoact2, builders._train_abc):
        script = build({"gpus": ""})[1]
        assert "CUDA_VISIBLE_DEVICES=0 " in script
        assert "--nproc-per-node=1 " in script


def test_train_refuses_an_impossible_gpu_count(monkeypatch):
    from yam_abc_reproduce.gui import builders, gpus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(gpus, "inventory", list)
    with pytest.raises(ValueError, match="only 1 GPU"):
        builders._train_abc({"gpus": "4"})
    with pytest.raises(ValueError, match="at least 1 GPU"):
        builders._train_abc({"gpus": "0"})


def test_deploy_commands_pin_one_gpu(monkeypatch):
    """A policy server needs one card, but XLA_PYTHON_CLIENT_MEM_FRACTION reserves 90% of
    every card it can see -- which is what blocks the next training launch. The banner has to
    be there too: the Deploy panel has no command box, so it is the only place the operator
    can learn which card the server took."""
    from yam_abc_reproduce.gui import builders, gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpus, "inventory", lambda: [("0", 500), ("1", 31000), ("5", 32000)])
    for cmd in (builders._deploy_openpi("/ck", "go", 8000, {}, "pi0_yam_lora"),
                builders._train_deploy_molmoact2("/ck", "go", 8202, {}),
                builders._train_deploy_abc("/ck", "go", 8300, {})):
        assert "CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID " in cmd
        assert "[yam-abc] pinned to GPU(s) 5" in cmd


def test_one_unreadable_card_does_not_disable_ranking(monkeypatch):
    """nvidia-smi prints [N/A] for a card it cannot read. Dropping the whole inventory over
    one bad row would silently revert every job to card 0 *and* switch the free-VRAM guard
    off -- worse than the single-column read this replaced."""
    from yam_abc_reproduce.gui import gpus

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpus.subprocess, "run", _fake_nvidia_smi(
        "0, GPU-a, 500\n1, GPU-b, [N/A]\n2, GPU-c, 32000\n"))
    assert gpus.inventory() == [("0", 500), ("1", None), ("2", 32000)]
    assert gpus.pick(1) == "2"          # still ranks the seven good cards
    assert gpus.pick(2) == "0,2"        # the unreadable card ranks below priced ones
    assert gpus.most_free_mib() == 32000


def test_unreadable_vram_does_not_look_like_a_full_card(monkeypatch):
    """An unpriced card must read as "no reading", not "0 MiB free". Scoring it 0 would make
    the guard refuse every launch on an idle box -- e.g. a driver hiccup where nvidia-smi
    exits non-zero with empty stdout while the operator has CUDA_VISIBLE_DEVICES=0 set."""
    from yam_abc_reproduce.gui import gpus

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(gpus.subprocess, "run", _fake_nvidia_smi(""))
    assert gpus.inventory() == [("0", None)]
    assert gpus.most_free_mib() is None  # guard skips; it must not 409 on a phantom 0 MiB
    assert gpus.pick(1) == "0"           # the offered card is still usable


def test_builder_refusal_reaches_the_operator(client):
    """app.js reads `detail` off the JSON body. An uncaught ValueError is a text/plain 500
    whose r.json() rejects, so the Launch button would silently do nothing.

    Uses `convert`, the one job kind that skips the free-VRAM guard, so this needs no GPU.
    """
    r = client.post("/api/jobs", json={"kind": "convert", "params": {}})
    assert r.status_code == 400
    assert "task" in r.json()["detail"]


def test_openpi_resume_and_overwrite_are_exclusive(no_gpus):
    """openpi's TrainConfig.__post_init__ rejects --resume with --overwrite, so ticking
    Resume used to kill the job at config parse."""
    from yam_abc_reproduce.gui import builders

    resumed = builders._train_openpi({"dataset": "", "resume": "1"}, "pi0_yam")[1]
    assert "--resume" in resumed and "--overwrite" not in resumed
    assert "--overwrite" in builders._train_openpi({"dataset": ""}, "pi0_yam")[1]


def test_abc_launch_refuses_without_a_prepared_cache(no_gpus, monkeypatch, tmp_path):
    """abc is the one backend that trains off a prepared cache instead of the dataset the form
    names, so a launch with no cache used to die inside validate_train_config -- i.e. only after
    torchrun had spun up every rank, with the real reason buried under a ChildFailedError."""
    from yam_abc_reproduce.gui import builders

    cache = tmp_path / "abc_cache"
    monkeypatch.setattr(builders, "_ABC_CACHE", cache)
    monkeypatch.setattr(builders, "_require_backend_venv", lambda backend: None)

    with pytest.raises(ValueError) as e:
        builders.build_train_command({"backend": "abc"})
    # Names every path train.py would have complained about, and how to produce them.
    for want in ("norm_stats.json", "train_real", "val_real", "export_mcap.py",
                 "compute_abc_norm_stats.py"):
        assert want in str(e.value)

    for name in ("train_real", "val_real"):
        (cache / name).mkdir(parents=True)
    (cache / "norm_stats.json").write_text("{}")
    assert "--mixture-preset=yam_abc " in builders.build_train_command({"backend": "abc"})[1]


def test_train_tab_shows_the_abc_cache_prerequisite(client):
    """The Train form cannot show this by itself -- abc's `task` field is vestigial, so the
    form looks complete while the one real prerequisite is invisible."""
    note = client.get("/api/train/fields?backend=abc").json()["note"]
    assert "data/abc_cache" in note and "export_mcap.py" in note
    assert client.get("/api/train/fields?backend=pi0").json()["note"] == ""


# --- convert progress -------------------------------------------------------
# yam-abc-convert emits no @metric lines, so the Train tab's bar has to be driven off the
# files it writes. The two formats need different counters: abc writes one .mcap per episode,
# while LeRobot v3 writes aggregates and only updates meta/info.json when an episode commits.

def _fake_corpus(tmp_path, frames, role="top", flag=True):
    """A source tree whose float64 <role>-timestamp.npy files imply `frames` per episode.

    Writes write_complete.flag last, as EpisodeRecorder does -- the converter only processes
    flagged dirs, so the progress denominator has to agree.
    """
    from yam_abc_reproduce.data.schema import WRITE_COMPLETE_FLAG

    src = tmp_path / "episodes" / "fold"
    src.mkdir(parents=True, exist_ok=True)
    base = len(list(src.glob("*")))     # a second call extends the corpus, as recording does
    for i, n in enumerate(frames):
        d = src / f"ep{base + i:04d}"
        d.mkdir()
        (d / f"{role}-timestamp.npy").write_bytes(b"\0" * (128 + 8 * n))
        if flag:
            (d / WRITE_COMPLETE_FLAG).write_text("")
    return src


def test_convert_progress_counts_frames_not_episodes(tmp_path, monkeypatch):
    """Takes range from tens to thousands of frames, so an episode count is a poor bar.
    88 + 3981 of 4069 total frames is 100% of the frames but only 2/3 of the episodes."""
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [88, 3981, 4000])
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_LEROBOT_HOME", tmp_path / "lerobot")
    info = tmp_path / "lerobot" / "ds" / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text('{"total_episodes": 2, "total_frames": 4069, "fps": 30}')

    s = cp.snapshot({"src": str(src), "to": "lerobot", "repo_id": "ds"}, None, time.time() - 100)
    assert (s["episodes_done"], s["episodes_total"]) == (2, 3)
    assert (s["frames_done"], s["frames_total"]) == (4069, 8069)
    assert abs(s["frac"] - 4069 / 8069) < 1e-6      # frames drive the bar, not 2/3
    assert s["eta_s"] > 0


def test_convert_progress_counts_mcaps_for_abc(tmp_path, monkeypatch):
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [100, 200, 300, 400])
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_ABC_OUT", tmp_path / "abc")
    out = tmp_path / "abc" / "ds"
    out.mkdir(parents=True)
    for i in range(3):
        (out / f"episode_{i:06d}.mcap").write_bytes(b"\x89MCAP0\r\n")

    s = cp.snapshot({"src": str(src), "to": "abc", "repo_id": "ds"}, None, time.time() - 60)
    assert (s["episodes_done"], s["episodes_total"]) == (3, 4)
    assert abs(s["frac"] - 0.75) < 1e-6


def test_convert_progress_survives_a_missing_or_partial_dataset(tmp_path, monkeypatch):
    """The endpoint is polled from the first second, before the converter has written
    anything -- and info.json is rewritten in place, so a read can catch it mid-write."""
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [10, 20])
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_LEROBOT_HOME", tmp_path / "lerobot")

    s = cp.snapshot({"src": str(src), "to": "lerobot", "repo_id": "nope"}, None, time.time())
    assert s["frames_done"] == 0 and s["frac"] == 0.0 and s["eta_s"] is None

    info = tmp_path / "lerobot" / "half" / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text('{"total_episodes": 1, "total_fram')   # truncated mid-write
    s = cp.snapshot({"src": str(src), "to": "lerobot", "repo_id": "half"}, None, time.time())
    assert s["episodes_done"] == 1 and s["frames_done"] == 0


def test_convert_progress_ignores_a_previous_runs_output(tmp_path, monkeypatch):
    """A killed abc run leaves its .mcap files behind (ABCFormat.begin only mkdirs), so a
    plain glob would report instant progress and freeze the bar there for the whole rerun."""
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [100] * 8)
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_ABC_OUT", tmp_path / "abc")
    out = tmp_path / "abc" / "ds"
    out.mkdir(parents=True)
    for i in range(8):                                  # leftovers from the killed run
        (out / f"episode_{i:06d}.mcap").write_bytes(b"x")

    started = time.time() + 1     # every existing file predates this job
    s = cp.snapshot({"src": str(src), "to": "abc", "repo_id": "ds"}, None, started)
    assert s["episodes_done"] == 0 and s["frac"] == 0.0


def test_convert_progress_rereads_the_source_each_poll(tmp_path, monkeypatch):
    """Caching the denominator by path went stale as soon as an episode was recorded or
    pruned between two conversions in one GUI session -- it read "4/3 episodes" at 100%."""
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [1000, 1000, 1000])
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_LEROBOT_HOME", tmp_path / "lerobot")
    info = tmp_path / "lerobot" / "fold" / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text('{"total_episodes": 1, "total_frames": 1000}')
    args = ({"src": str(src), "to": "lerobot", "repo_id": "fold"}, None, time.time() - 60)
    assert cp.snapshot(*args)["frames_total"] == 3000

    _fake_corpus(tmp_path, [1000] * 6)                  # operator records 6 more takes
    assert cp.snapshot(*args)["frames_total"] == 9000    # 3 + 6, not the cached 3000


def test_convert_progress_skips_unflagged_source_dirs(tmp_path, monkeypatch):
    """The converter only processes dirs holding write_complete.flag. A take stranded by a
    hard kill would otherwise inflate the denominator so the bar never reached 100%."""
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [100, 100, 100])
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_LEROBOT_HOME", tmp_path / "lerobot")
    stranded = src / "20260739_killed"                  # recorder.start() leftover, no flag
    stranded.mkdir()
    (stranded / "top-timestamp.npy").write_bytes(b"\0" * (128 + 8 * 700))

    info = tmp_path / "lerobot" / "fold" / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text('{"total_episodes": 3, "total_frames": 300}')
    s = cp.snapshot({"src": str(src), "to": "lerobot", "repo_id": "fold"}, None, time.time() - 60)
    assert (s["episodes_total"], s["frames_total"]) == (3, 300)
    assert s["frac"] == 1.0 and s["eta_s"] is None       # a finished job reads as finished


def test_convert_progress_without_a_top_camera(tmp_path, monkeypatch):
    """The camera roster is configurable -- a station can run left+wrist with no `top`.
    Hardcoding top-timestamp.npy pinned the bar at 0% for the whole run on such a station."""
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [300, 1200], role="wrist")
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_LEROBOT_HOME", tmp_path / "lerobot")
    info = tmp_path / "lerobot" / "fold" / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text('{"total_episodes": 1, "total_frames": 300}')

    s = cp.snapshot({"src": str(src), "to": "lerobot", "repo_id": "fold"}, None, time.time() - 60)
    assert s["frames_total"] == 1500 and s["frac"] > 0 and s["eta_s"] is not None


def test_convert_progress_prefers_metadata_num_frames(tmp_path, monkeypatch):
    """metadata.json is the recorder's own count and the converter's reference timeline;
    prefer it over inferring from a timestamp file's size."""
    from yam_abc_reproduce.gui import convert_progress as cp

    src = _fake_corpus(tmp_path, [100])
    (src / "ep0000" / "metadata.json").write_text('{"num_frames": 4242, "task_name": "t"}')
    monkeypatch.setattr(cp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(cp, "_LEROBOT_HOME", tmp_path / "lerobot")
    s = cp.snapshot({"src": str(src), "to": "lerobot", "repo_id": "none"}, None, time.time())
    assert s["frames_total"] == 4242


def test_convert_progress_endpoint_rejects_a_non_convert_job(client):
    """Both 404 arms: a job of the wrong kind, and an id that does not exist. The first needs
    a real job registered -- POSTing kind=train just gets refused on a box with no backend."""
    from yam_abc_reproduce.gui.jobs import Job

    r = client.get("/api/jobs/nope-1/convert-progress")
    assert r.status_code == 404 and "convert" in r.json()["detail"]

    jobs = client.app.state.jobs
    jobs._jobs["train-1"] = Job(id="train-1", kind="train", params={}, real_command="echo hi")
    r = client.get("/api/jobs/train-1/convert-progress")
    assert r.status_code == 404, r.text
    assert "convert" in r.json()["detail"]


def test_estop(client):
    client.post("/api/collect/start-teleop")
    r = client.post("/api/estop").json()
    assert r["estopped"] is True
    assert client.get("/api/collect/status").json()["estopped"] is True
