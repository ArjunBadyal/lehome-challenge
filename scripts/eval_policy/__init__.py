"""
LeHome Challenge Policy Module

This module provides the base policy interface and implementations
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
from .lerobot_policy import LeRobotPolicy
from .sac_policy import SacPolicy
from .example_participant_policy import CustomPolicy
from .residual_policy import ResidualPolicy
from .hierarchical_residual_policy import HierarchicalResidualPolicy
from .encoder_residual_policy import EncoderResidualPolicy
from .router_policy import RouterPolicy
from .policy_stabilizer import (
    PolicyStabilizer,
    StabilizedLeRobotPolicy,
    StabilizedRouterPolicy,
)
from .scripted_collar_recovery_policy import ScriptedCollarRecoveryPolicy
from .vision_collar_recovery_policy import VisionCollarRecoveryPolicy
from .ai_teleop_policy import AITelop_TopShort
from .cem_recovery_policy import CEMRecoveryTopShort
from .portfolio_router_policy import PortfolioRouterPolicy
from .seen_garment_router_policy import SeenGarmentRouterPolicy
from .submission_bundle_policy import SubmissionBundlePolicy
from .docker_policy import DockerPolicy

__all__ = [
    "BasePolicy",
    "PolicyRegistry",
    "LeRobotPolicy",
    "SacPolicy",
    "CustomPolicy",
    "ResidualPolicy",
    "HierarchicalResidualPolicy",
    "EncoderResidualPolicy",
    "RouterPolicy",
    "PolicyStabilizer",
    "StabilizedLeRobotPolicy",
    "StabilizedRouterPolicy",
    "ScriptedCollarRecoveryPolicy",
    "VisionCollarRecoveryPolicy",
    "SeenGarmentRouterPolicy",
    "SubmissionBundlePolicy",
]
