"""
Trajectory Calibration Package.

Post-hoc uncertainty calibration and multi-scale trajectory signatures
for Multimodal Large Language Models (M3-LLaVA & MQT-LLaVA).
"""

__version__ = "0.1.0"

from trajectory_calibration import calibrators, features, metrics, uq, utils

try:
    from trajectory_calibration import vlm
except ImportError:
    vlm = None

__all__ = ["calibrators", "features", "metrics", "uq", "utils", "vlm"]
