"""HTTP client adapter for testing a running Docker policy container locally.

Talks to the upstream Docker policy server contract documented in
submission/UPSTREAM_README.md (`POST /reset`, `POST /infer`).
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry


def _serialize_observation(obs: Dict[str, np.ndarray]) -> dict:
    out = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            out[key] = {
                "base64": base64.b64encode(value.tobytes()).decode("ascii"),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, np.ndarray):
            out[key] = value.astype(np.float32).tolist()
        else:
            out[key] = value
    return out


@PolicyRegistry.register("docker")
class DockerPolicy(BasePolicy):
    def __init__(self, model_path=None, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.url = os.environ.get("LEHOME_DOCKER_URL", "http://127.0.0.1:8080")
        self._action_queue: List[np.ndarray] = []
        self._wait_ready()

    def _wait_ready(self, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._post("/reset", {})
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(1)
        raise RuntimeError(f"Docker policy server at {self.url} not ready in {timeout}s")

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

    def reset(self):
        self._action_queue.clear()
        self._post("/reset", {})

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if not self._action_queue:
            payload = _serialize_observation(observation)
            response = self._post("/infer", payload)
            actions = response.get("actions", [])
            if not actions:
                raise RuntimeError("Docker policy returned no actions")
            self._action_queue.extend(np.asarray(a, dtype=np.float32) for a in actions)
        return self._action_queue.pop(0)
