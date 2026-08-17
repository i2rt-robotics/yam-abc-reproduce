"""Client that talks to a policy server over the openpi websocket protocol.

Thin wrapper over ``openpi_client.WebsocketClientPolicy`` that adds:

  * open-loop action chunking -- query the server only every
    ``open_loop_horizon`` control steps and replay the cached chunk in
    between, so a slow VLA (100-300 ms/chunk) still drives a 30 Hz loop.
  * optional client-side image resize -- shrink frames before send to cut
    bandwidth/latency (off by default; the server also resizes to whatever the
    model wants, so leaving this off never loses information).

``openpi_client`` is the lightweight client package
(``third_party/policy/openpi/packages/openpi-client``); it pulls in only
``websockets``/``msgpack``/``numpy`` -- no torch or jax -- so this stays
installable in the YAM-ABC-Reproduce robot-host env.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np


def _bypass_proxy_for(host: str) -> None:
    """Exempt the policy server from any proxy set in the environment.

    websockets >= 14 proxies by default and urllib does not exempt loopback, so
    ``ws://127.0.0.1`` is routed at the proxy and fails as if the server were down. A
    policy server is always reached directly. Both spellings: either can win urllib's scan.
    """
    for var in ("no_proxy", "NO_PROXY"):
        hosts = [h.strip() for h in os.environ.get(var, "").split(",") if h.strip()]
        if host not in hosts:
            os.environ[var] = ",".join([*hosts, host])


@runtime_checkable
class PolicyClient(Protocol):
    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """Return one ``(S,)`` action row for this control step."""
        ...

    def reset(self) -> None:
        """Drop any cached chunk so the next call re-queries the server."""
        ...


class WebsocketPolicyClient:
    """Open-loop-chunked websocket policy client.

    Args:
        host/port: policy server address.
        open_loop_horizon: how many rows to execute from a chunk before
            re-querying. Default 15 to match ABC's execute_chunk_dim; smaller =
            more reactive + more server load, larger = smoother + more lag.
        resize: optional ``(H, W)`` to resize every image to before sending
            (uses openpi_client.image_tools.resize_with_pad). ``None`` sends
            full resolution and lets the server resize.
        api_key: optional bearer key if the server requires auth.
    """

    def __init__(
        self,
        host: str,
        port: int,
        open_loop_horizon: int = 15,
        resize: tuple[int, int] | None = None,
        api_key: str | None = None,
        rtc: bool = False,
        rtc_prefix_length: int = 4,
        rtc_action_horizon: int = 15,
        rtc_lead_steps: int = 4,
        smooth: int = 4,
    ) -> None:
        from openpi_client import websocket_client_policy

        _bypass_proxy_for(host)  # WebsocketClientPolicy passes no proxy=None of its own
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port, api_key)
        self.metadata: dict = self._client.get_server_metadata()
        self.open_loop_horizon = int(open_loop_horizon)
        self._resize = resize
        # RTC (real-time chunking): condition each new chunk on the *tail* of the
        # current one (where the plan ends) so the trajectory continues forward.
        # Mirrors the reference async_lbm_agent scheme. Off by default (pi0/
        # molmoact2 use plain open-loop chunking).
        self.rtc = bool(rtc)
        self.rtc_prefix_length = int(rtc_prefix_length)   # P: rows frozen as the prefix
        self.rtc_action_horizon = int(rtc_action_horizon)  # H: rows executed per chunk
        self.rtc_lead_steps = int(rtc_lead_steps)          # L: re-query when L rows remain
        # Cross-chunk smoothing: linearly blend the first ``smooth`` rows of a new
        # chunk with the previous chunk's un-executed continuation, so re-query
        # boundaries don't snap. 0 disables. (matches the reference deployment.)
        self.smooth = int(smooth)
        self._chunk: np.ndarray | None = None
        self._pending: np.ndarray | None = None
        self._idx = 0
        # Async RTC: a background thread re-queries the server while the control
        # loop streams the buffer at full rate, so inference latency never stalls
        # the loop (mirrors the reference async_lbm_agent). Guarded by _lock.
        self._lock = threading.Lock()
        self._latest_obs: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _maybe_resize(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._resize is None:
            return obs
        from openpi_client import image_tools

        h, w = self._resize
        images = {
            role: image_tools.resize_with_pad(np.asarray(img), h, w)
            for role, img in obs["images"].items()
        }
        return {**obs, "images": images}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Raw one-shot inference: send obs, return the server's full response
        dict (``{"actions": (H, S), ...}``). Bypasses chunk caching."""
        return self._client.infer(self._maybe_resize(obs))

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """Return one action row, re-querying the server at chunk boundaries."""
        if self.rtc:
            return self._get_action_rtc(obs)
        if self._chunk is None or self._idx >= min(self.open_loop_horizon, len(self._chunk)):
            resp = self.infer(obs)
            actions = np.array(resp["actions"], dtype=np.float32)  # copy: msgpack arrays are read-only
            if actions.ndim == 1:  # server returned a single row
                actions = actions[None, :]
            # Blend the new chunk's start with the old chunk's un-executed tail so
            # the arm doesn't snap at the re-query boundary.
            if self.smooth > 0 and self._chunk is not None:
                m = min(self.smooth, len(actions), max(0, len(self._chunk) - self._idx))
                if m > 0:
                    w = np.linspace(1.0 / m, 1.0, m).reshape(-1, 1)
                    cont = self._chunk[self._idx : self._idx + m]
                    actions[:m] = w * actions[:m] + (1.0 - w) * cont
            self._chunk = actions
            self._idx = 0
        row = self._chunk[self._idx]
        self._idx += 1
        return row

    def _rtc_query(self, obs: dict[str, Any], prefix: np.ndarray) -> np.ndarray:
        """Query with a P-row action prefix; the server freezes those as the new
        chunk's first ``P`` rows and generates the continuation. Drop the frozen
        overlap and keep the next ``H`` rows -- the executable continuation."""
        P, H = self.rtc_prefix_length, self.rtc_action_horizon
        actions = np.array(
            self.infer({**obs, "action_prefix": prefix, "prefix_length": P})["actions"],
            dtype=np.float32,
        )
        if actions.ndim == 1:
            actions = actions[None, :]
        return actions[P : P + H]

    def _get_action_rtc(self, obs: dict[str, Any]) -> np.ndarray:
        """Async real-time chunking (matches the reference async_lbm_agent scheme).

        The control loop calls this at its full rate and only ever *reads* a row
        from the current chunk buffer -- it never blocks on inference. A background
        thread (`_rtc_loop`) re-queries the server conditioned on the *tail* of the
        current chunk (its last ``P`` rows = where the plan ends) and queues the
        continuation; the executor swaps to it when the current chunk is exhausted.
        Conditioning on the plan's endpoint -- not the current pose -- carries
        motion forward so the arm commits instead of creeping, and running the
        query off-loop keeps the arm streaming at full rate."""
        with self._lock:
            self._latest_obs = obs
        # First chunk: seed synchronously (one-time ~150 ms) from the current
        # state tiled P times, then hand off to the background thread.
        if self._thread is None:
            P = self.rtc_prefix_length
            state = np.asarray(obs["state"], dtype=np.float32).reshape(1, -1)
            first = self._rtc_query(obs, np.repeat(state, P, axis=0))
            with self._lock:
                self._chunk, self._pending, self._idx = first, None, 0
            self._stop.clear()
            self._thread = threading.Thread(target=self._rtc_loop, daemon=True)
            self._thread.start()
        with self._lock:
            chunk = self._chunk
            if self._idx >= len(chunk):
                # Current chunk exhausted: swap to the queued continuation if the
                # background thread has it ready, else hold the last row (underrun).
                if self._pending is not None:
                    self._chunk = chunk = self._pending
                    self._pending = None
                    self._idx = 0
                else:
                    self._idx = len(chunk) - 1
            row = np.array(chunk[self._idx], dtype=np.float32)
            self._idx += 1
        return row

    def _rtc_loop(self) -> None:
        """Background thread: keep a continuation chunk queued. When the executor
        has consumed down to the last ``L`` rows and no continuation is pending,
        re-query using the current chunk's tail and stash it as ``_pending``."""
        H, P, L = self.rtc_action_horizon, self.rtc_prefix_length, self.rtc_lead_steps
        while not self._stop.is_set():
            with self._lock:
                need = (
                    self._chunk is not None
                    and self._pending is None
                    and self._idx >= max(1, H - L)
                )
                tail = self._chunk[-P:].copy() if need else None
                obs = self._latest_obs
            if need and obs is not None:
                try:
                    cont = self._rtc_query(obs, tail)
                except Exception:
                    time.sleep(0.01)
                    continue
                with self._lock:
                    if self._pending is None:  # still wanted (no reset raced us)
                        self._pending = cont
            else:
                time.sleep(0.002)

    def reset(self) -> None:
        # Stop the background RTC thread (if any) before clearing state, so a
        # re-seed on the next get_action starts a fresh thread cleanly.
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            self._chunk = None
            self._pending = None
            self._idx = 0
            self._latest_obs = None
        self._client.reset()
