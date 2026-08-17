"""FastAPI app factory for the unified collect / train / deploy GUI.

A thin shell: REST + a 5 Hz WebSocket status feed + per-camera JPEG previews +
static frontend. All robot/training logic lives behind CollectSession and
JobManager, never in the routes.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import (
    CAMERA_ROLE_OPTIONS,
    CAMERA_TYPE_OPTIONS,
    CAN_CHANNEL_OPTIONS,
    CONTROLLER_TYPE_OPTIONS,
    FORMAT_OPTIONS,
    GRIPPER_OPTIONS,
    ROBOT_TYPE_OPTIONS,
    StationConfig,
    apply_station_form,
    build_station_config,
    controller_channel,
    load_yaml,
    robot_channel,
    save_cameras_yaml,
)
from ..data import video
from ..robot.can_bus import list_can_interfaces
from . import convert_progress, gpus
from .jobs import JobManager
from .schemas import CreateJob, DeployStart, StartRecording, StationForm, ZeroGello
from .session import CollectSession

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    cfg: StationConfig,
    mock: bool = False,
    station_path: str | None = None,
    cameras_path: str | None = None,
) -> FastAPI:
    app = FastAPI(title="YAM-ABC-Reproduce", version=__version__)
    session = CollectSession(cfg, mock=mock)
    jobs = JobManager()
    _transcode_lock = threading.Lock()  # serializes review-video H.264 transcodes

    def base_cfg() -> StationConfig:
        """Fresh config for the next connect/go-live. Re-reads the station YAML so
        bench calibration edits (signs/offsets/gripper) apply without restarting the
        process; falls back to the last-known config if the path is unset or the file
        is momentarily unparseable."""
        if station_path is None:
            return session.cfg
        try:
            return build_station_config(station_path, cameras_path)
        except Exception:
            return session.cfg

    def cameras_file() -> Path | None:
        """The cameras.yaml the rail's Preview writes back to: the explicit
        ``--cameras`` path, else the station file's ``cameras_config:``. None when
        the roster is inlined in the station file (nothing safe to overwrite)."""
        if cameras_path is not None:
            return Path(cameras_path)
        if station_path is None:
            return None
        try:
            raw = load_yaml(station_path)
        except Exception:
            return None
        cp = raw.get("cameras_config")
        return Path(cp) if cp else None

    _gpu_cache = {"t": 0.0, "v": None}

    def _gpu_status():
        """utilization/memory from nvidia-smi, cached ~2s (the WS feed is 5 Hz)."""
        import subprocess as _sp
        import time as _time
        now = _time.time()
        if now - _gpu_cache["t"] > 2.0:
            _gpu_cache["t"] = now
            try:
                out = _sp.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2).stdout
                u, m, t = [int(x.strip()) for x in out.strip().splitlines()[0].split(",")]
                _gpu_cache["v"] = {"util_pct": u, "mem_used_mib": m,
                                   "mem_total_mib": t, "mem_pct": round(100 * m / t)}
            except Exception:
                _gpu_cache["v"] = None
        return _gpu_cache["v"]

    def full_status() -> dict:
        s = session.status()
        s["jobs"] = [j.summary() for j in jobs.list()]
        s["gpu"] = _gpu_status()
        return s

    # --- health -----------------------------------------------------------
    @app.get("/api/health")
    def health():
        return {"version": __version__, "mock": mock, "task": session.cfg.task_name}

    # --- station config (drives the editable left config rail) ------------
    @app.get("/api/config")
    def config():
        c = session.cfg
        return {
            # `channel` is the operator's override (None = derive from type); `channel_default`
            # is the bus the type resolves to, shown as the placeholder for a blank field.
            "robots": [
                {
                    "type": r.type,
                    "gripper": r.gripper,
                    "channel": r.channel,
                    "channel_default": robot_channel(r.type),
                }
                for r in c.robot.robots
            ],
            "controllers": [
                {
                    "type": ct.type,
                    "controls": ct.controls,
                    "channel": ct.channel,
                    "channel_default": controller_channel(ct.type),
                }
                for ct in c.robot.controllers
            ],
            "cameras": [
                {"name": cm.name, "role": cm.role, "type": cm.type, "serial": cm.serial}
                for cm in c.cameras
            ],
            "control_hz": c.control_hz,
            "task_name": c.task_name,
            "data_format": c.data_format,
            "save_root": c.save_root,
            # Station default for the Deploy tab's "home pose" field (None = don't home).
            "deploy_home_pose": c.deploy_home_pose,
            "mock": mock,
            # Dropdown option sets so the rail can render selects without hardcoding.
            "options": {
                "controller_type": CONTROLLER_TYPE_OPTIONS,
                "robot_type": ROBOT_TYPE_OPTIONS,
                "gripper": GRIPPER_OPTIONS,
                "camera_type": CAMERA_TYPE_OPTIONS,
                "camera_role": CAMERA_ROLE_OPTIONS,
                "data_format": FORMAT_OPTIONS,
                # CAN interfaces present on this machine, plus the standard udev names so
                # the picker still works before the buses are up.
                "can_channel": sorted(set(list_can_interfaces()) | set(CAN_CHANNEL_OPTIONS)),
            },
        }

    # --- collect ----------------------------------------------------------
    @app.get("/api/collect/status")
    def collect_status():
        return full_status()

    @app.post("/api/collect/connect")
    def connect_cameras(form: StationForm | None = Body(default=None)):
        """Open the cameras for preview (no robot/CAN/teleop). Applies the rail's
        camera config so previews reflect edits; called on Collect-tab load and by
        the Preview button."""
        base = base_cfg()
        cfg_to_use = apply_station_form(base, form.model_dump()) if form is not None else base
        try:
            session.connect_cameras(cfg_to_use)
        except RuntimeError as e:  # e.g. camera-set change refused while live
            raise HTTPException(status_code=409, detail=str(e))
        if form is not None:
            cf = cameras_file()
            if cf is not None:
                try:
                    save_cameras_yaml(cf, cfg_to_use.cameras)
                except OSError:
                    pass  # best-effort; preview still works this session
        return session.status()

    @app.post("/api/collect/start-teleop")
    def start_teleop(form: StationForm | None = Body(default=None)):
        # First call goes live: apply the edited rail config, bring CAN up, build
        # devices, enable sync. Subsequent calls just re-enable sync.
        base = base_cfg()
        try:
            cfg_to_use = apply_station_form(base, form.model_dump()) if form is not None else base
        except ValueError as e:  # e.g. two devices assigned the same CAN bus
            raise HTTPException(status_code=400, detail=str(e))
        try:
            session.start_teleop(cfg_to_use)
        except RuntimeError as e:  # e.g. live for autonomy — needs a Reset Session first
            raise HTTPException(status_code=409, detail=str(e))
        return session.status()

    @app.post("/api/collect/stop-teleop")
    def stop_teleop():
        session.stop_teleop()
        return session.status()

    @app.post("/api/collect/start-recording")
    def start_recording(body: StartRecording):
        try:
            path = session.start_recording(body.task_name)
        except RuntimeError as e:  # recorder not built yet (before go-live)
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:  # empty task name
            raise HTTPException(status_code=400, detail=str(e))
        return {"episode": path}

    @app.post("/api/collect/stop-recording")
    def stop_recording():
        try:
            return session.stop_recording()
        except RuntimeError as e:  # not live / not recording
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/api/collect/episodes")
    def episodes():
        eps = session.list_episodes()
        return {"count": len(eps), "episodes": eps}

    # --- episode review (post-collection sanity check) --------------------
    from ..data.schema import WRITE_COMPLETE_FLAG

    def _review_roots() -> dict[str, Path]:
        """Episode sources the Review tab browses. Both use the same on-disk format;
        episodes are nested by task (and, for rollouts, policy):
          dataset: data/episodes/<task>/<ep>     rollout: data/rollouts/<policy>/<task>/<ep>"""
        return {
            "dataset": Path(session.cfg.save_root),
            "rollout": Path("data/rollouts"),
        }

    def _episode_dir(ep_id: str) -> Path:
        """Resolve a *completed* episode dir across the review roots, rejecting
        traversal. ``ep_id`` is the path relative to its root (may include the
        task/policy folders, e.g. ``pi0/pick_bottle/20260711_ab12``)."""
        for root in _review_roots().values():
            root = root.resolve()
            d = (root / ep_id).resolve()
            if (root == d or root in d.parents) and (d / WRITE_COMPLETE_FLAG).exists():
                return d
        raise HTTPException(status_code=404, detail="episode not found")

    def _episode_meta(d: Path) -> dict:
        mp = d / "metadata.json"
        try:
            return json.loads(mp.read_text()) if mp.exists() else {}
        except Exception:
            return {}

    @app.get("/api/review/episodes")
    def review_episodes():
        """List completed episodes (newest first) with light metadata for the picker,
        scanning both roots recursively. Each episode carries its source
        (dataset/rollout) and the task (+ policy for rollouts) derived from its path,
        so the UI can filter by them. ``id`` is the path relative to its root."""
        out = []
        for source, root in _review_roots().items():
            root = root.resolve()
            if not root.exists():
                continue
            for flag in root.rglob(WRITE_COMPLETE_FLAG):
                d = flag.parent
                rel = d.relative_to(root)
                parts = rel.parts  # rollout: <policy>/<task>/<ep>; dataset: <task>/<ep>
                m = _episode_meta(d)
                task = m.get("task_name") or (parts[-2] if len(parts) >= 2 else None)
                policy = parts[0] if (source == "rollout" and len(parts) >= 3) else None
                out.append(
                    {
                        "id": str(rel),
                        "source": source,
                        "policy": policy,
                        "task": task,
                        "task_name": m.get("task_name"),
                        "num_frames": m.get("num_frames"),
                        "created_at": m.get("created_at"),
                        "arm_names": m.get("arm_names", []),
                        "cameras": [c.get("role") for c in m.get("cameras", [])],
                    }
                )
        out.sort(key=lambda e: e.get("created_at") or e["id"], reverse=True)
        return {"episodes": out}

    @app.delete("/api/review/{ep_id:path}")
    def review_delete(ep_id: str):
        """Delete an episode directory (traversal-safe: only within a review root)."""
        import shutil

        d = _episode_dir(ep_id)  # raises 404 if not a completed episode in a root
        shutil.rmtree(d)
        return {"deleted": ep_id}

    @app.get("/api/review/{ep_id:path}/meta")
    def review_meta(ep_id: str):
        return _episode_meta(_episode_dir(ep_id))

    @app.get("/api/review/{ep_id:path}/video/{role}")
    def review_video(ep_id: str, role: str):
        d = _episode_dir(ep_id)
        safe = "".join(ch for ch in role if ch.isalnum() or ch in "_-")
        src = d / f"{safe}-images-rgb.mp4"
        if not src.exists():
            raise HTTPException(status_code=404, detail="video not found")
        # The recorder writes H.264, which a <video> tag plays as-is. Episodes recorded
        # before that change are MPEG-4 Part 2, which browsers cannot play, so those (and
        # only those) get a cached H.264 copy alongside them.
        if video.video_codec_name(src) == "h264":
            return FileResponse(str(src), media_type="video/mp4")
        h264 = d / f"{safe}-images-rgb.h264.mp4"
        stale = (not h264.exists()) or (h264.stat().st_mtime < src.stat().st_mtime)
        if stale:
            with _transcode_lock:
                if (not h264.exists()) or (h264.stat().st_mtime < src.stat().st_mtime):
                    tmp = h264.with_suffix(".tmp.mp4")
                    try:
                        video.transcode_to_h264(src, tmp)
                        tmp.replace(h264)
                    except Exception:
                        tmp.unlink(missing_ok=True)
                        # Fall back to the original — may not play, but better than 500.
                        return FileResponse(str(src), media_type="video/mp4")
        return FileResponse(str(h264), media_type="video/mp4")

    @app.get("/api/review/{ep_id:path}/signals")
    def review_signals(ep_id: str):
        """Per-arm state + action arrays and per-camera timestamps as JSON, for the
        synced curve overlay. Episodes are short (~hundreds of steps) so no downsampling."""
        import numpy as np

        d = _episode_dir(ep_id)
        m = _episode_meta(d)

        def load(name: str):
            p = d / f"{name}.npy"
            return np.load(p).tolist() if p.exists() else None

        arms = m.get("arm_names", [])
        data: dict = {
            "num_frames": m.get("num_frames"),
            "control_hz": m.get("control_hz"),
            "arm_names": arms,
            "arms": {},
            "cameras": {},
        }
        for a in arms:
            data["arms"][a] = {
                "joint_pos": load(f"{a}-joint_pos"),
                "gripper_pos": load(f"{a}-gripper_pos"),
                "action_joint": load(f"action-{a}-joint"),
                "action_gripper": load(f"action-{a}-gripper"),
            }
        for c in m.get("cameras", []):
            role = c.get("role")
            data["cameras"][role] = {"timestamps": load(f"{role}-timestamp")}
        return data

    # --- cameras ----------------------------------------------------------
    @app.get("/api/cameras")
    def cameras():
        return {"cameras": session.camera_descriptors()}

    @app.get("/api/cameras/detect")
    def detect_cameras():
        """Live-hardware scan so the rail can offer real serials to pick from
        (instead of hand-editing cameras.yaml per machine)."""
        from ..camera.discovery import discover_cameras

        return {"devices": discover_cameras()}

    @app.get("/api/cameras/{name}/preview.jpg")
    def preview(name: str, eye: str | None = None):
        jpg = session.hub.preview_jpeg(name, eye=eye)
        if jpg is None:
            return Response(status_code=404)
        return Response(content=jpg, media_type="image/jpeg")

    # --- train field schema (drives the adaptive Train form) --------------
    @app.get("/api/train/fields")
    def train_fields(backend: str = "pi0"):
        from .train_schema import BACKENDS, fields_for, note_for

        return {"backends": BACKENDS, "backend": backend, "fields": fields_for(backend),
                "note": note_for(backend)}


    # --- policy-server lifecycle helpers -----------------------------------
    def _port_probe(host: str, port: int, timeout: float = 0.6) -> bool:
        """Is something accepting TCP on host:port?

        Closes politely -- with a real request, and only after reading the reply. A
        websockets server logs a full handshake-failure traceback for every connection
        that hangs up early, and this polls every few seconds. Reading matters: hanging
        up mid-response produces the same traceback.
        """
        import socket as _socket
        try:
            with _socket.create_connection((host, port), timeout=timeout) as s:
                try:
                    s.sendall(
                        b"GET /healthz HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n"
                        % host.encode()
                    )
                    s.recv(1)
                except OSError:
                    pass  # listening is all we came to learn; the courtesy is best-effort
                return True
        except OSError:
            return False

    def _local_port_owner(port: int):
        """Best-effort (pid, cmdline) of the LISTEN holder of :port via ss."""
        import re as _re, subprocess as _sp
        from pathlib import Path as _Path
        try:
            out = _sp.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=3).stdout
        except Exception:
            return None, None
        for line in out.splitlines():
            if _re.search(rf"[:\]]{port}\s", line) and "LISTEN" in line:
                m = _re.search(r"pid=(\d+)", line)
                if not m:
                    return None, ""
                pid = int(m.group(1))
                try:
                    cmd = _Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ").strip()
                except OSError:
                    cmd = ""
                return pid, cmd
        return None, None

    # --- jobs (train / deploy) ----------------------------------------
    @app.post("/api/jobs")
    def create_job(body: CreateJob):
        # Guard rails for "Start Server (local)": refuse a busy port or a GPU mostly
        # held by someone else, with actionable messages.
        if body.kind == "deploy":
            # Reject a pi0 checkpoint under the pi05 backend (and vice versa) up front,
            # matching on the "pi05" marker in the checkpoint path.
            backend = str(body.params.get("backend") or "")
            ckpt = str(body.params.get("checkpoint") or "")
            if backend in ("pi0", "pi05") and ckpt:
                is05 = "pi05" in ckpt
                if backend == "pi05" and not is05 and "pi0" in ckpt:
                    raise HTTPException(status_code=409, detail=(
                        f"backend is pi05 but the checkpoint looks like a pi0 one ({ckpt}). "
                        "Pick a pi05_* checkpoint from the dropdown."))
                if backend == "pi0" and is05:
                    raise HTTPException(status_code=409, detail=(
                        f"backend is pi0 but the checkpoint looks like a pi05 one ({ckpt}). "
                        "Pick a pi0_* checkpoint from the dropdown."))
            port = int(body.params.get("port") or 8000)
            if _port_probe("127.0.0.1", port):
                pid, cmd = _local_port_owner(port)
                who = f" by pid {pid} ({(cmd or '')[:90]})" if pid else ""
                raise HTTPException(status_code=409, detail=(
                    f"port {port} is already in use{who} -- a policy server may "
                    "already be running. Click Stop Server to kill it, or just "
                    "Load & Run against the existing one."))
        if body.kind in ("deploy", "train"):
            # Ask whether *any* card can take the job, not whether card 0 can: builders.py
            # pins the job to the freest one, so guarding card 0 would refuse a launch in
            # exactly the case the auto-pick exists for.
            free = gpus.most_free_mib()
            if free is not None and free < 8000:
                raise HTTPException(status_code=409, detail=(
                    f"no GPU has 8000 MiB free (most free: {free} MiB) -- another process is "
                    "holding every card (click the GPU badge to inspect/kill it). "
                    "Free one first or the model load will OOM."))
        try:
            job = jobs.launch(body.kind, body.params)
        except ValueError as e:
            # Builder refusals: missing dataset, backend not installed, more GPUs than exist.
            # These have to be an HTTPException -- app.js reads `detail` off the JSON body,
            # and an uncaught ValueError is a text/plain 500 whose r.json() rejects, so the
            # Launch button would just do nothing.
            raise HTTPException(status_code=400, detail=str(e)) from e
        return job.summary()

    @app.get("/api/deploy/server-status")
    def deploy_server_status(host: str = "127.0.0.1", port: int = 8000):
        """Probe the policy server: ready (accepting TCP) / loading (job running,
        not listening yet) / error (job exited non-zero) / none."""
        local = host in ("127.0.0.1", "localhost", "0.0.0.0", "")
        listening = _port_probe("127.0.0.1" if local else host, port)
        pid = cmd = None
        if local:
            pid, cmd = _local_port_owner(port)
        last = None
        for j in jobs.list():
            sm = j.summary()
            if sm.get("kind") == "deploy":
                last = sm
        if listening:
            state = "ready"
        elif last and last.get("status") == "running":
            state = "loading"
        elif last and last.get("status") == "exited" and last.get("returncode") not in (0, None):
            state = "error"
        else:
            state = "none"
        return {"state": state, "listening": listening, "pid": pid,
                "cmd": (cmd or "")[:120] or None, "job": last}

    @app.post("/api/deploy/server-stop")
    def deploy_server_stop(port: int = 8000):
        """Stop the local policy server: the GUI-launched job AND/OR any external
        process listening on :port whose cmdline looks like one of our servers."""
        import os as _os, signal as _sig, time as _time
        n = jobs.stop_kind("deploy")
        killed = []
        denied = None
        pid, cmd = _local_port_owner(port)
        if pid and any(k in (cmd or "") for k in
                       ("openpi_server", "molmoact_server", "abc_server")):
            try:
                _os.kill(pid, _sig.SIGTERM)
                for _ in range(20):
                    if not _port_probe("127.0.0.1", port, 0.2):
                        break
                    _time.sleep(0.25)
                else:
                    _os.kill(pid, _sig.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                denied = f"pid {pid} is owned by another user; run on the server:  sudo kill {pid}"
        return {"stopped_jobs": n, "killed": killed, "port": port, "denied": denied}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.summary() for j in jobs.list()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        return job.summary() if job else Response(status_code=404)

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(job_id: str, since: int = 0):
        lines, cursor = jobs.logs(job_id, since)
        return {"lines": lines, "cursor": cursor}

    @app.get("/api/jobs/{job_id}/metrics")
    def job_metrics(job_id: str, since: int = 0):
        pts, cursor = jobs.metrics(job_id, since)
        return {"points": pts, "cursor": cursor}

    @app.get("/api/jobs/{job_id}/convert-progress")
    def job_convert_progress(job_id: str):
        """Live progress for a convert job, read off the files it writes.

        yam-abc-convert emits no @metric lines, so the Train tab's progress bar has nothing
        to draw from during a conversion -- this fills that gap. Cheap enough to poll at 1 Hz:
        one small read per source episode, one info.json, and the kernel's own write counters.
        No du over the staging tree, which holds tens of thousands of PNGs mid-episode.
        """
        job = jobs.get(job_id)
        if job is None or job.kind != "convert":
            raise HTTPException(status_code=404, detail=f"no convert job {job_id!r}")
        return convert_progress.snapshot(job.params, job.pid, job.started)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str):
        return {"stopped": jobs.stop(job_id)}

    # --- deploy: scan trained checkpoints for the ckpt-path dropdown -------
    @app.get("/api/deploy/checkpoints")
    def deploy_checkpoints(backend: str = "pi0"):
        """List trained checkpoint paths for `backend`, newest first, so the
        Deploy tab can offer a dropdown instead of a blind text field."""
        from .builders import _ABC_CACHE, _MOLMOACT, _OPENPI
        found: list[tuple[float, str]] = []
        note = None
        try:
            if backend in ("pi0", "pi05"):
                base = _OPENPI / "checkpoints"
                if base.is_dir():
                    for cfg in base.iterdir():
                        ok = (cfg.name.startswith("pi05") if backend == "pi05"
                              else cfg.name.startswith("pi0") and not cfg.name.startswith("pi05"))
                        if not (ok and cfg.is_dir()):
                            continue
                        for exp in cfg.iterdir():
                            if not exp.is_dir():
                                continue
                            for step in exp.iterdir():
                                if (step / "params").is_dir():
                                    found.append((step.stat().st_mtime,
                                                  str(step.relative_to(_OPENPI))))
            elif backend == "molmoact2":
                base = _MOLMOACT / "experiments" / "checkpoints" / "finetune"
                if base.is_dir():
                    for run in base.iterdir():
                        if run.is_dir():
                            for step in run.glob("step*"):
                                if step.is_dir():
                                    found.append((step.stat().st_mtime, str(step)))
                    if found:
                        note = "OLMo-core checkpoints — convert to HF (convert_molmoact2_to_hf.py) before deploy"
            elif backend == "abc":
                base = _ABC_CACHE / "finetune_checkpoints"
                if base.is_dir():
                    for f in base.glob("*.pt"):
                        found.append((f.stat().st_mtime, str(f)))
        except OSError:
            pass
        found.sort(reverse=True)
        return {"checkpoints": [p for _, p in found[:50]], "note": note}

    # --- camera health: status / holders / reset ---------------------------
    def _video_holders():
        """Processes holding /dev/video* fds (own-user pids always visible)."""
        import os as _os
        import pwd as _pwd
        from pathlib import Path as _Path
        holders = {}
        for pd in _Path("/proc").iterdir():
            if not pd.name.isdigit():
                continue
            try:
                for l in (pd / "fd").iterdir():
                    try:
                        tgt = _os.readlink(l)
                    except OSError:
                        continue
                    if tgt.startswith("/dev/video"):
                        pid = int(pd.name)
                        if pid not in holders:
                            st = pd.stat()
                            try:
                                cmd = (pd / "cmdline").read_text().replace("\0", " ").strip()[:120]
                            except OSError:
                                cmd = ""
                            holders[pid] = {"pid": pid, "user": _pwd.getpwuid(st.st_uid).pw_name,
                                            "devs": set(), "cmd": cmd, "self": pid == _os.getpid()}
                        holders[pid]["devs"].add(tgt)
            except (PermissionError, FileNotFoundError):
                continue
        out = []
        for h in holders.values():
            h["devs"] = sorted(h["devs"])
            out.append(h)
        return out

    @app.get("/api/cameras/health")
    def cameras_health(full: bool = False):
        """Camera health. Default (light) mode reads only in-memory worker state —
        safe to poll. ``full=1`` additionally enumerates RealSense devices over
        USB and scans /proc for /dev/video holders; USB enumeration disturbs the
        CAN adapters sharing the bus (polling it caused a 5s teleop stutter), so
        it runs only on demand (opening the popup)."""
        import os as _os
        present = {}
        if full:
            try:
                import pyrealsense2 as rs
                for d in rs.context().query_devices():
                    present[d.get_info(rs.camera_info.serial_number)] = d.get_info(rs.camera_info.name)
            except Exception:
                pass
        live = {getattr(w, "name", None) for w in session.workers}
        cams = []
        for cm in base_cfg().cameras:
            cams.append({"name": cm.name, "role": cm.role, "type": cm.type, "serial": cm.serial,
                         "detected": (cm.serial in present) if (full and cm.type == "realsense") else None,
                         "streaming": cm.name in live})
        return {"cameras": cams, "holders": (_video_holders() if full else []),
                "gui_pid": _os.getpid(), "full": full}

    @app.post("/api/cameras/reset")
    def cameras_reset():
        """Recover wedged cameras: stop workers, hardware-reset every RealSense,
        re-open. Refused while the robot session is live (would yank frames
        mid-teleop)."""
        import time as _time
        if session.status().get("live"):
            raise HTTPException(status_code=409, detail="session is live — stop teleop/deploy first")
        try:
            session._disconnect_cameras()
        except Exception:
            pass
        n = 0
        try:
            import pyrealsense2 as rs
            for d in rs.context().query_devices():
                try:
                    d.hardware_reset()
                    n += 1
                except Exception:
                    pass
        except Exception:
            pass
        _time.sleep(6 if n else 1)
        err = None
        try:
            session.connect_cameras(base_cfg())
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        return {"reset_devices": n,
                "streaming": [getattr(w, "name", "?") for w in session.workers],
                "error": err}

    @app.post("/api/cameras/kill-holder")
    def cameras_kill_holder(pid: int):
        import os as _os
        import signal as _sig
        import time as _time
        hs = {h["pid"]: h for h in _video_holders()}
        if pid not in hs:
            raise HTTPException(status_code=404, detail=f"pid {pid} holds no /dev/video*")
        if pid == _os.getpid():
            raise HTTPException(status_code=400, detail="that is the GUI itself — use Reset Cameras instead")
        try:
            _os.kill(pid, _sig.SIGTERM)
        except PermissionError:
            raise HTTPException(status_code=403, detail=(
                f"no permission (user {hs[pid]['user']}). Run on the server:  sudo kill {pid}"))
        except ProcessLookupError:
            return {"killed": pid, "note": "already gone"}
        for _ in range(12):
            try:
                _os.kill(pid, 0)
            except ProcessLookupError:
                return {"killed": pid}
            _time.sleep(0.25)
        try:
            _os.kill(pid, _sig.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
        return {"killed": pid, "note": "SIGKILL"}

    # --- GPU processes: inspect / kill squatters ---------------------------
    @app.get("/api/gpu/procs")
    def gpu_procs():
        import os as _os
        import pwd as _pwd
        import subprocess as _sp
        from pathlib import Path as _Path
        procs = []
        try:
            out = _sp.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=3).stdout
        except Exception:
            out = ""
        me = _os.getuid()
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            try:
                pid_s, mem_s = [x.strip() for x in line.split(",")[:2]]
                pid, mem = int(pid_s), int(mem_s)
            except ValueError:
                continue
            try:
                st = _os.stat(f"/proc/{pid}")
                own = st.st_uid == me
                user = _pwd.getpwuid(st.st_uid).pw_name
                cmd = _Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ").strip()[:140]
            except OSError:
                own, user, cmd = False, "?", ""
            procs.append({"pid": pid, "user": user, "mem_mib": mem, "cmd": cmd, "own": own})
        return {"procs": procs, "self_pid": _os.getpid()}

    @app.post("/api/gpu/kill")
    def gpu_kill(pid: int):
        """Kill a process currently using the GPU. Only pids from /api/gpu/procs
        are accepted; the GUI itself is protected; other users' pids 403 with a
        copy-pasteable sudo hint."""
        import os as _os
        import signal as _sig
        import time as _time
        listed = {p["pid"]: p for p in gpu_procs()["procs"]}
        if pid not in listed:
            raise HTTPException(status_code=404, detail=f"pid {pid} is not on the GPU (already gone?)")
        if pid == _os.getpid():
            raise HTTPException(status_code=400, detail="refusing to kill the GUI itself")
        try:
            _os.kill(pid, _sig.SIGTERM)
        except PermissionError:
            raise HTTPException(status_code=403, detail=(
                f"no permission to kill pid {pid} (user {listed[pid]['user']}). "
                f"Run on the server:  sudo kill {pid}"))
        except ProcessLookupError:
            return {"killed": pid, "note": "already gone"}
        for _ in range(12):
            try:
                _os.kill(pid, 0)
            except ProcessLookupError:
                return {"killed": pid}
            _time.sleep(0.25)
        try:
            _os.kill(pid, _sig.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
        return {"killed": pid, "note": "SIGKILL"}

    # --- maintenance ------------------------------------------------------
    @app.post("/api/maintenance/reset-can")
    def reset_can():
        """Bring the CAN buses back up (e.g. after re-plugging an arm)."""
        from ..robot.can_bus import reset_can_buses

        ok, out = reset_can_buses()
        return {"ok": ok, "output": out}

    @app.post("/api/maintenance/zero-gello")
    def zero_gello(body: ZeroGello):
        """Hardware-zero a passive-GELLO leader at its current pose (writes the
        encoder EEPROM). Hold the leader at the follower's home pose first."""
        return session.zero_gello(body.side)

    @app.post("/api/maintenance/end-hardware-session")
    def end_hardware_session():
        """Release robot control while keeping the browser service available."""
        session.estop()
        deploy_jobs_stopped = jobs.stop_kind("deploy")
        return {"ending": True, "deploy_jobs_stopped": deploy_jobs_stopped,
                "note": "hardware released; web service remains available"}

    @app.post("/api/maintenance/power-off-arms")
    def power_off_arms():
        """Stop control and physically disable the follower motors."""
        deploy_jobs_stopped = jobs.stop_kind("deploy")
        result = session.power_off_arms()
        result["deploy_jobs_stopped"] = deploy_jobs_stopped
        return result

    # --- graceful shutdown: release hardware on SIGTERM/SIGINT -------------
    # Killing the GUI without this leaves RealSense pipelines wedged (previews
    # 404 until a hardware reset) and, mid-session, motor loops running. Uvicorn
    # turns SIGTERM/SIGINT into this shutdown event.
    @app.on_event("shutdown")
    def _graceful_shutdown():
        import logging as _logging
        _logging.getLogger("yam_abc_reproduce").info("shutdown: releasing robot + cameras")
        print("[yam-abc] shutdown: stopping loops and releasing cameras...", flush=True)
        for step in (session.estop, session._disconnect_cameras):
            try:
                step()
            except Exception:  # noqa: BLE001
                pass

    # --- safety -----------------------------------------------------------
    @app.post("/api/estop")
    def estop():
        # Stop the live loop AND any running deploy job. Safety-critical.
        session.estop()
        n = jobs.stop_kind("deploy")
        return {"estopped": True, "deploy_jobs_stopped": n}

    @app.post("/api/session/reset")
    def reset_session():
        """Recover after an E-STOP (or a failed go-live) without restarting the GUI:
        tear down the loops + robot units, reset the CAN buses, clear the estop latch.
        The next Start Teleop rebuilds cleanly. Cameras/previews are kept."""
        return session.reset_session()

    # --- autonomous deploy: in-session policy client ----------------------
    @app.post("/api/deploy/start")
    def deploy_start(body: DeployStart):
        # The server (a Job via /api/jobs) is separate; this drives the client
        # loop over the previewing cameras + built robots against host:port.
        # Surface failures (server unreachable, missing deps, CAN down) to the UI
        # instead of a bare 500 — the client runs in-session with no job log.
        try:
            session.start_deploy(
                host=body.host, port=body.port, prompt=body.prompt,
                cfg=base_cfg(), open_loop_horizon=body.open_loop_horizon,
                record=body.record, save_root=body.save_root, home_pose=body.home_pose,
                rtc=body.rtc, rtc_prefix_length=body.rtc_prefix_length,
                rtc_action_horizon=body.rtc_action_horizon,
                rtc_lead_steps=body.rtc_lead_steps,
                max_joint_speed=body.max_joint_speed,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
        return {"deploying": True, "recording": body.record}

    @app.post("/api/deploy/stop")
    def deploy_stop():
        saved = session.stop_deploy()
        return {"deploying": False, "saved": saved}

    # --- websocket status feed (~5 Hz) ------------------------------------
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(full_status())
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            pass

    # Static frontend mounted last so API routes take precedence.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    app.state.session = session
    app.state.jobs = jobs
    return app
