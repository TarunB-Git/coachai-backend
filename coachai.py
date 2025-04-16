import cv2
import mediapipe as mp
import numpy as np
import argparse

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- Offline Processing Functions ---
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            frame_keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        else:
            frame_keypoints = None
        keypoints.append(frame_keypoints)
    cap.release()
    return keypoints


def calculate_difference(kp1, kp2):
    diffs = []
    for a, b in zip(kp1, kp2):
        if a and b:
            diffs.append(np.linalg.norm(np.array(a) - np.array(b)))
        else:
            diffs.append(None)
    return diffs


def compare_videos(teacher_kps, student_kps):
    frame_differences = []
    for t_kp, s_kp in zip(teacher_kps, student_kps):
        if t_kp and s_kp:
            frame_differences.append(calculate_difference(t_kp, s_kp))
    return frame_differences


def summarize_differences(differences):
    if not differences:
        print("No valid frame comparisons found.")
        return [], ["No movement data available for analysis. Check video quality or pose detection."]
    avg_diffs = np.nanmean(differences, axis=0)
    if not isinstance(avg_diffs, np.ndarray):
        print("Average differences is not an array. Something went wrong.")
        return [], ["Pose data may be incomplete."]
    tips = []
    try:
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_WRIST.value] > 0.1:
            tips.append("Try keeping your right hand steadier.")
        if avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > 0.1:
            tips.append("Work on your left knee position during lunges.")
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_KNEE.value] > 0.1:
            tips.append("Focus on maintaining right knee alignment.")
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_ANKLE.value] > 0.1:
            tips.append("Your right foot positioning needs more stability.")
        if avg_diffs[mp_pose.PoseLandmark.LEFT_ELBOW.value] > 0.1:
            tips.append("Watch your left elbow movement.")
        if avg_diffs[mp_pose.PoseLandmark.LEFT_WRIST.value] > 0.1:
            tips.append("Keep your left hand under better control.")
    except IndexError as e:
        print("Indexing error in tip generation:", e)
        tips.append("Some body landmarks may be missing or undetected.")
    # Heuristic accuracy score
    valid_diffs = avg_diffs[~np.isnan(avg_diffs)]
    mean_diff = np.mean(valid_diffs)
    accuracy_score = max(0, 100 - (mean_diff * 100))
    print(f"Movement Accuracy Score: {accuracy_score:.2f}%")
    return avg_diffs, tips


def visualize_comparison(teacher_video, student_video, output_path="output_comparison.mp4", threshold=0.1):
    cap1 = cv2.VideoCapture(teacher_video)
    cap2 = cv2.VideoCapture(student_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1280, 480))
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        res1 = pose.process(img1)
        res2 = pose.process(img2)
        if res1.pose_landmarks and res2.pose_landmarks:
            kp1 = [(lm.x, lm.y, lm.z) for lm in res1.pose_landmarks.landmark]
            kp2 = [(lm.x, lm.y, lm.z) for lm in res2.pose_landmarks.landmark]
            diffs = calculate_difference(kp1, kp2)
            mp_drawing.draw_landmarks(frame1, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame2, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for i, d in enumerate(diffs):
                if d and d > threshold:
                    h, w = frame1.shape[:2]
                    p1 = res1.pose_landmarks.landmark[i]
                    p2 = res2.pose_landmarks.landmark[i]
                    cv2.circle(frame1, (int(p1.x*w), int(p1.y*h)), 5, (0, 0, 255), -1)
                    cv2.circle(frame2, (int(p2.x*w), int(p2.y*h)), 5, (0, 0, 255), -1)
        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)
    cap1.release()
    cap2.release()
    out.release()
    print(f"Comparison video saved to {output_path}")

# --- Real-Time Functions ---
def load_teacher_reference(video_path):
    cap = cv2.VideoCapture(video_path)
    ref_kp = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if res.pose_landmarks:
            ref_kp = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            break
    cap.release()
    return ref_kp


def live_pose_feedback(teacher_kps, threshold=0.1):
    cv2.namedWindow('Live Pose Feedback', cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam. Check device connection.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: No frame received from webcam.")
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        if res.pose_landmarks and teacher_kps:
            student_kps = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            # Compute 2D differences
            diffs2d = []
            for i, (t, s) in enumerate(zip(teacher_kps, student_kps)):
                if t and s:
                    dx = t[0] - s[0]
                    dy = t[1] - s[1]
                    diffs2d.append((i, np.hypot(dx, dy)))
            if diffs2d:
                mean_diff2d = np.mean([d for _, d in diffs2d])
                accuracy = max(0, 100 - mean_diff2d * 100)
            else:
                accuracy = 0
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for idx, d in diffs2d:
                if d > threshold:
                    h, w = frame.shape[:2]
                    lm = res.pose_landmarks.landmark[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)
        cv2.imshow('Live Pose Feedback', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# --- Accuracy Scoring Function ---
def compute_accuracy_score(differences, threshold=0.1):
    scores = []
    for frame_diff in differences:
        valid_diffs = [d for d in frame_diff if d is not None]
        if not valid_diffs:
            scores.append(0)
            continue
        match_count = sum(1 for d in valid_diffs if d <= threshold)
        scores.append((match_count / len(valid_diffs)) * 100)
    return scores

# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description='CoachAI: Pose Comparison and Real-Time Feedback')
    parser.add_argument('--teacher', required=True, help='Path to teacher video')
    parser.add_argument('--student', help='Path to student video (omit for live mode)')
    parser.add_argument('--live', action='store_true', help='Enable real-time webcam feedback')
    parser.add_argument('--threshold', type=float, default=0.1, help='Highlight threshold for differences (normalized)')
    parser.add_argument('--output', default='output_comparison.mp4', help='Output path for comparison video')
    args = parser.parse_args()
    if args.live:
        teacher_kps = load_teacher_reference(args.teacher)
        if not teacher_kps:
            print("Failed to load teacher reference keypoints.")
            return
        live_pose_feedback(teacher_kps, threshold=args.threshold)
    else:
        if not args.student:
            print("Please provide a student video in non-live mode.")
            return
        teacher_kps = extract_keypoints(args.teacher)
        student_kps = extract_keypoints(args.student)
        print(f"Teacher frames with keypoints: {sum(kp is not None for kp in teacher_kps)} / {len(teacher_kps)}")
        print(f"Student frames with keypoints: {sum(kp is not None for kp in student_kps)} / {len(student_kps)}")
        differences = compare_videos(teacher_kps, student_kps)
        avg_diffs, tips = summarize_differences(differences)
        print("Average Differences:", avg_diffs)
        print("Tips for Improvement:", tips)
        visualize_comparison(args.teacher, args.student, output_path=args.output, threshold=args.threshold)
        accuracy_scores = compute_accuracy_score(differences, threshold=args.threshold)
        avg_accuracy = np.mean(accuracy_scores)
        print(f"Overall Pose Accuracy: {avg_accuracy:.2f}%")

if __name__ == "__main__":
    main()