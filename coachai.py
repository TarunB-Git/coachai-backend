import argparse

import numpy as np

from coachai_backend.core import JOINT_NAMES, run_offline_session
from coachai_backend.realtime import live_pose_feedback, load_teacher_reference


def main():
    parser = argparse.ArgumentParser(description="CoachAI: Pose Comparison and Real-Time Feedback")
    parser.add_argument("--teacher", required=True, help="Path to teacher video")
    parser.add_argument("--student", help="Path to student video (omit for live mode)")
    parser.add_argument("--live", action="store_true", help="Enable real-time webcam feedback")
    parser.add_argument("--threshold", type=float, default=0.1, help="Highlight threshold for differences (normalized)")
    parser.add_argument("--sport", default="general", help="Sport context: fencing, skating, or general")
    args = parser.parse_args()

    if args.live:
        teacher_kps = load_teacher_reference(args.teacher)
        if not teacher_kps:
            print("Failed to load teacher reference keypoints.")
            return
        live_pose_feedback(teacher_kps, threshold=args.threshold, sport=args.sport)
        return

    if not args.student:
        print("Please provide a student video in non-live mode.")
        return

    result = run_offline_session(
        teacher_video=args.teacher,
        student_video=args.student,
        threshold=args.threshold,
        sport=args.sport,
    )

    avg_diffs = result["average_differences"]
    print("Average Differences:", avg_diffs)
    print("Tips for Improvement:", result["tips"])
    print(f"Overall Pose Accuracy: {result['average_accuracy']:.2f}%")

    if isinstance(avg_diffs, np.ndarray):
        for idx, value in enumerate(avg_diffs):
            if idx < len(JOINT_NAMES):
                _ = (JOINT_NAMES[idx], value)


if __name__ == "__main__":
    main()
