"""CoachAI backend package."""

from .core import (
    JOINT_NAMES,
    compare_videos,
    extract_keypoints,
    summarize_differences,
    visualize_comparison,
    compute_accuracy_score,
)
from .analytics import export_differences_to_csv, plot_average_errors, plot_joint_errors_over_time
from .realtime import load_teacher_reference, live_pose_feedback

__all__ = [
    "JOINT_NAMES",
    "compare_videos",
    "extract_keypoints",
    "summarize_differences",
    "visualize_comparison",
    "compute_accuracy_score",
    "export_differences_to_csv",
    "plot_average_errors",
    "plot_joint_errors_over_time",
    "load_teacher_reference",
    "live_pose_feedback",
]
