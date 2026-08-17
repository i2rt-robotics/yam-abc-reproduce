"""
MolmoAct2 inference clients.

MolmoActClientBase — ABC interface (schema, state_adapter, action_adapter).
_MolmoActHTTPClient — shared HTTP + chunk-buffering implementation.
DroidClient / YAMClient — concrete embodiment clients.

Adding a new embodiment: subclass _MolmoActHTTPClient, set the three
class attributes (schema, state_adapter, action_adapter). Done.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .common import (
    MOLMOACT2_SCHEMAS,
    droid_state_adapter,
    yam_state_adapter,
    yam_action_adapter,
    extract_camera,
    extract_qpos,
)

logger = logging.getLogger(__name__)


class MolmoActClientBase(ABC):
    """Interface for a MolmoAct2 inference client.

    Subclasses must declare:

        schema         = MOLMOACT2_SCHEMAS["my_embodiment"]
        state_adapter  = my_state_adapter   # or None
        action_adapter = my_action_adapter  # or None
    """

    @property
    @abstractmethod
    def schema(self): ...

    @property
    @abstractmethod
    def state_adapter(self): ...

    @property
    @abstractmethod
    def action_adapter(self): ...

    @abstractmethod
    def infer(self, obs: dict, instruction: str) -> np.ndarray:
        """Return the next action for obs and instruction."""

    @abstractmethod
    def reset(self) -> None:
        """Clear buffered actions. Call at each episode boundary."""


class _MolmoActHTTPClient(MolmoActClientBase):
    """Shared HTTP + chunk-buffering implementation."""

    schema         = None
    state_adapter  = None
    action_adapter = None

    def __init__(
        self,
        url: str,
        *,
        n_action_steps: Optional[int] = None,
        request_timeout: float = 60.0,
    ) -> None:
        try:
            import requests
            import json_numpy
            json_numpy.patch()
            self._session = requests.Session()
        except ImportError as e:
            raise ImportError(
                "Clients require `requests` and `json-numpy`: "
                "pip install requests json-numpy"
            ) from e

        self.url = url
        self.n_action_steps = int(n_action_steps) if n_action_steps is not None else None
        self.request_timeout = request_timeout
        self._queue: list[np.ndarray] = []

        logger.info("%s ready | url=%s  cameras=%s",
                    type(self).__name__, url, list(self.schema.camera_keys))

    def infer(self, obs: dict, instruction: str) -> np.ndarray:
        if not self._queue:
            chunk = self._query_server(obs, instruction)
            n = self.n_action_steps if self.n_action_steps is not None else len(chunk)
            self._queue = list(chunk[:max(1, n)])
        raw = np.asarray(self._queue.pop(0), dtype=np.float32)
        if self.action_adapter is not None:
            return np.asarray(self.action_adapter(raw), dtype=np.float32)
        return raw

    def reset(self) -> None:
        self._queue.clear()

    def _query_server(self, obs: dict, instruction: str) -> list[np.ndarray]:
        import json_numpy

        qpos = extract_qpos(obs)
        if self.state_adapter is not None:
            qpos = np.asarray(self.state_adapter(qpos), dtype=np.float32)

        payload: dict = {"instruction": instruction, "state": qpos}
        for cam_key in self.schema.camera_keys:
            payload[cam_key] = extract_camera(obs, cam_key)

        t0 = time.time()
        resp = self._session.post(
            self.url,
            headers={"Content-Type": "application/json"},
            data=json_numpy.dumps(payload),
            timeout=self.request_timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Server error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        logger.debug("infer %.3fs  dt_ms=%s", time.time() - t0, data.get("dt_ms"))

        actions = np.asarray(
            data["actions"] if isinstance(data, dict) and "actions" in data else data
        )
        if actions.ndim == 1:
            actions = actions[None, :]
        return [np.asarray(a) for a in actions]


class DroidClient(_MolmoActHTTPClient):
    """MolmoAct2-DROID client (external_cam + wrist_cam, 8-D state)."""
    schema         = MOLMOACT2_SCHEMAS["droid"]
    state_adapter  = staticmethod(droid_state_adapter)
    action_adapter = None


class YAMClient(_MolmoActHTTPClient):
    """MolmoAct2-YAM client (top_cam + left_cam + right_cam, 14-D state)."""
    schema         = MOLMOACT2_SCHEMAS["yam"]
    state_adapter  = staticmethod(yam_state_adapter)
    action_adapter = staticmethod(yam_action_adapter)
