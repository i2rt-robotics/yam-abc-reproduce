"""Job / JobManager: launch a subprocess, stream its stdout into a ring buffer,
and stop it. Backs both the Train and Deploy tabs.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

from .builders import build_command

_MAX_LOG_LINES = 5000
# Training/eval processes emit progress as ``@metric {"step":N,"loss":..,...}`` lines
# on stdout; the runner parses these into a series the GUI plots. Backend-agnostic:
# any trainer that prints this line format gets a live loss curve for free.
METRIC_PREFIX = "@metric "

# Fallback for trainers we don't emit @metric from (molmoact2's OLMo-core loop, abc):
# scrape their native "... step N ... loss 0.42 ..." stdout into metric points. Gated
# to those backends so pi0's own @metric lines aren't double-counted.
# NOTE: no ``\b`` before ``loss`` — molmoact2 prints ``train/action_flow_loss=1.33``
# where ``loss`` is preceded by ``_`` (a word char), so ``\bloss`` never matches.
_LOSS_RE = re.compile(r"loss[\s=:/]*=?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", re.I)
_STEP_RE = re.compile(r"\bstep[\s=:]+([0-9]+)", re.I)
_SCRAPE_BACKENDS = {"molmoact2", "abc"}


def _scrape_loss(job: "Job", line: str) -> None:
    """Best-effort: pull a loss (and step) out of a native trainer log line into a
    metric point. molmoact2's OLMo-core loop prints the step and the loss on
    *separate* lines (``[step=18/20, …]`` then ``train/action_flow_loss=1.33``), so
    we latch the most recent step onto the job and attach it to the next loss line."""
    sm = _STEP_RE.search(line)
    if sm:
        job._scrape_step = int(sm.group(1))  # type: ignore[attr-defined]
    m = _LOSS_RE.search(line)
    if not m:
        return
    try:
        loss = float(m.group(1))
    except ValueError:
        return
    step = getattr(job, "_scrape_step", None)
    if step is None:
        step = job.metrics[-1]["step"] + 1 if job.metrics else 1
    if job.metrics and job.metrics[-1].get("step") == step:
        job.metrics[-1]["loss"] = loss  # same step: keep latest loss, no dup point
        return
    job.metrics.append({"step": step, "loss": loss})


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the process's whole group, not just the direct child. Train jobs are
    ``bash -lc 'uv run …'`` which fork uv/torchrun/git children; without this they
    orphan on stop. Requires the child to lead its own group (start_new_session)."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    real_command: str
    status: str = "pending"  # pending | running | exited
    returncode: int | None = None
    logs: list[str] = field(default_factory=list)
    log_dropped: int = 0  # lines front-trimmed from logs; keeps `since` cursors valid
    metrics: list[dict] = field(default_factory=list)  # parsed @metric points
    started: float = 0.0    # launch time; convert progress derives its rate from this
    _proc: subprocess.Popen | None = None
    _thread: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def summary(self) -> dict:
        last = self.metrics[-1] if self.metrics else None
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "returncode": self.returncode,
            "real_command": self.real_command,
            "log_lines": len(self.logs),
            "metric_points": len(self.metrics),
            "last_metric": last,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def launch(self, kind: str, params: dict) -> Job:
        cmd, real = build_command(kind, params)
        with self._lock:
            self._counter += 1
            job_id = f"{kind}-{self._counter}"
        job = Job(id=job_id, kind=kind, params=params, real_command=real)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # New session -> the child leads its own process group, so stop() can
            # signal the whole tree (bash -> uv -> torchrun/git/python).
            start_new_session=True,
        )
        job._proc = proc
        job.started = time.time()
        job.status = "running"
        job._thread = threading.Thread(target=self._pump, args=(job,), daemon=True)
        job._thread.start()
        with self._lock:
            self._jobs[job_id] = job
        return job

    def _pump(self, job: Job) -> None:
        assert job._proc is not None and job._proc.stdout is not None
        scrape = job.kind == "train" and job.params.get("backend") in _SCRAPE_BACKENDS
        for line in job._proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(METRIC_PREFIX):
                try:
                    job.metrics.append(json.loads(line[len(METRIC_PREFIX) :]))
                    continue  # keep metric lines out of the human log
                except Exception:
                    pass  # malformed -> fall through and show it as a log line
            elif scrape:
                _scrape_loss(job, line)
            job.logs.append(line)
            if len(job.logs) > _MAX_LOG_LINES:
                n_del = len(job.logs) - _MAX_LOG_LINES
                del job.logs[0:n_del]
                job.log_dropped += n_del
        job._proc.wait()
        job.returncode = job._proc.returncode
        job.status = "exited"

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def logs(self, job_id: str, since: int = 0) -> tuple[list[str], int]:
        job = self._jobs.get(job_id)
        if job is None:
            return [], since
        start = max(0, since - job.log_dropped)
        lines = job.logs[start:]
        return lines, job.log_dropped + len(job.logs)

    def metrics(self, job_id: str, since: int = 0) -> tuple[list[dict], int]:
        job = self._jobs.get(job_id)
        if job is None:
            return [], since
        pts = job.metrics[since:]
        return pts, since + len(pts)

    def stop(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job._proc is None:
            return False
        proc = job._proc
        if proc.poll() is None:
            _signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
        return True

    def stop_kind(self, kind: str) -> int:
        n = 0
        for job in list(self._jobs.values()):
            if job.kind == kind and job._proc is not None and job._proc.poll() is None:
                self.stop(job.id)
                n += 1
        return n
