"""Local eval adapter for the packaged submission policy.

This wrapper lets `scripts.eval` exercise the exact Docker-submission policy
implementation and `submission/policies/` artifacts without building Docker.
It is for validation only; the submitted runtime remains `submission/policy.py`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_ROOT = REPO_ROOT / "submission"


@PolicyRegistry.register("submission_bundle")
class SubmissionBundlePolicy(BasePolicy):
    """Adapter around `submission.policy.LeHomePolicy`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.environ.setdefault("POLICIES_ROOT", str(SUBMISSION_ROOT / "policies"))
        if str(SUBMISSION_ROOT) not in sys.path:
            sys.path.insert(0, str(SUBMISSION_ROOT))

        spec = importlib.util.spec_from_file_location(
            "lehome_submission_policy", SUBMISSION_ROOT / "policy.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load submission/policy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.policy = module.LeHomePolicy()

    def reset(self):
        self.policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        actions = self.policy.infer(observation)
        if not actions:
            raise RuntimeError("submission policy returned no actions")
        return np.asarray(actions[0], dtype=np.float32)
