import cv2
import mediapipe as mp
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
JOINT_NAMES = [lm.name for lm in mp_pose.PoseLandmark]

def generate_heuristics(avg_diffs, differences, sport="general", joint_threshold=0.1):
    """
    Generates a list of recommendations based on average errors per joint.
    
    Parameters:
      avg_diffs: 1D numpy array with average differences for each joint.
      differences: list of lists (frames x joints) with differences (for computing variance).
      sport: a string indicating the sport context ('fencing', 'skating', 'general')
      joint_threshold: baseline threshold for triggering a tip.
      
    Returns:
      A list of heuristic tips (strings).
    """
    tips = []
    
    # Calculate per-joint standard deviations, which can be used to gauge consistency.
    if differences and differences[0] is not None:
        variances = np.std(differences, axis=0)
    else:
        variances = np.zeros_like(avg_diffs)
    
    # Fencing specific heuristics
    if sport.lower() == "fencing":
        # Example: In fencing, the guard position is vital.
        # Check right wrist and right elbow (for a strong guard) as well as the torso alignment.
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_WRIST.value] > joint_threshold:
            tips.append("Focus on keeping your right wrist stable to maintain a strong guard.")
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_ELBOW.value] > joint_threshold * 0.8:
            tips.append("Improve the alignment of your right elbow for better parries.")
        # You might also consider hip or shoulder positions in a real case.
        # Add more rules based on expert advice.
    
    # Skating specific heuristics
    elif sport.lower() == "skating":
        # In skating, lower body stability is crucial.
        if avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > joint_threshold:
            tips.append("Work on stabilizing your left knee, as it’s key for balance on skates.")
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_ANKLE.value] > joint_threshold:
            tips.append("Your right ankle may be unstable; consider drills to improve foot control.")
        # You might add suggestions related to hip and core engagement.
    
    # General heuristics
    else:  # 'general' or any unspecified sport
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_WRIST.value] > joint_threshold:
            tips.append("Try keeping your right hand steadier.")
        if avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > joint_threshold:
            tips.append("Work on your left knee positioning for better stability.")
        if variances[mp_pose.PoseLandmark.RIGHT_WRIST.value] > 0.05:
            tips.append("High variability in your right wrist movement suggests a need for consistency drills.")
    
    # You can add more overall checks. For example, if any joint shows very high error:
    if np.any(avg_diffs > 0.15):
        tips.append("Some joints have very high deviations. Consider slow-motion practice to correct your form.")
    
    # A summary of overall performance can be added as a tip:
    overall_error = np.mean(avg_diffs)
    if overall_error < joint_threshold * 0.5:
        tips.append("Excellent overall movement consistency!")
    elif overall_error < joint_threshold:
        tips.append("Good performance, but there’s room for improvement in specific areas.")
    else:
        tips.append("Overall movement error is high; consider a thorough review of your technique.")

    return tips
    

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

import csv  # Ensure this is imported at the top

def export_differences_to_csv(differences, output_csv="differences.csv"):
    """
    Exports frame-by-frame differences to a CSV file.
    
    Each row represents one frame, with columns:
      frame, followed by a column for each joint (named according to JOINT_NAMES)
    """
    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header: frame index and joint names.
            writer.writerow(["frame"] + JOINT_NAMES)
            for i, frame_diff in enumerate(differences):
                # Format each difference to four decimal places; use empty string if None.
                row = [i] + [f"{d:.4f}" if d is not None else "" for d in frame_diff]
                writer.writerow(row)
        print(f"Exported differences to {output_csv}")
    except Exception as e:
        print("Error exporting CSV:", e)

def plot_average_errors(avg_diffs, output_path="avg_errors.png"):
    """
    Plots a bar chart of average error per joint.
    
    - avg_diffs: a 1D numpy array (output from np.nanmean on differences)
    - output_path: the filename where the chart will be saved.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_diffs)), avg_diffs, color="skyblue")
        plt.xticks(range(len(avg_diffs)), JOINT_NAMES, rotation=90)
        plt.xlabel("Joint")
        plt.ylabel("Average Error")
        plt.title("Average Error per Joint")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved average error chart to {output_path}")
    except Exception as e:
        print("Error plotting average errors:", e)

def plot_joint_errors_over_time(differences, joint_indices, output_prefix="joint_errors"):
    """
    Plots a line chart of error over time for the specified joints.
    
    - differences: list of lists containing per-frame differences (frames x joints)
    - joint_indices: list of integers representing the joint indices to plot
    - output_prefix: filename prefix; each file will be named as <output_prefix>_<joint name>.png
    """
    num_frames = len(differences)
    for idx in joint_indices:
        # Build a list of errors for joint 'idx' in each frame. Use NaN if not available.
        errors = [frame[idx] if frame[idx] is not None else np.nan for frame in differences]
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_frames), errors, marker="o", linestyle="-")
            plt.xlabel("Frame Index")
            plt.ylabel("Error")
            plt.title(f"Error over Time: {JOINT_NAMES[idx]}")
            plt.tight_layout()
            out_file = f"{output_prefix}_{JOINT_NAMES[idx]}.png"
            plt.savefig(out_file)
            plt.close()
            print(f"Saved time-series for {JOINT_NAMES[idx]} to {out_file}")
        except Exception as e:
            print(f"Error plotting errors for joint {JOINT_NAMES[idx]}:", e)



# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description='CoachAI: Pose Comparison and Real-Time Feedback')
    parser.add_argument('--teacher', required=True, help='Path to teacher video')
    parser.add_argument('--student', help='Path to student video (omit for live mode)')
    parser.add_argument('--live', action='store_true', help='Enable real-time webcam feedback')
    parser.add_argument('--threshold', type=float, default=0.1, help='Highlight threshold for differences (normalized)')
    parser.add_argument('--output', default='output_comparison.mp4', help='Output path for comparison video')
    parser.add_argument('--sport', default='general', help='Sport context: fencing, skating, or general')

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
        # After computing avg_diffs and obtaining basic tips from summarize_differences()
        avg_diffs, basic_tips = summarize_differences(differences)

        # Now generate sport-specific heuristics
        # For example, for fencing:
        sport_tips = generate_heuristics(avg_diffs, differences, sport=args.sport, joint_threshold=args.threshold)

        # Merge or choose one set of tips:
        tips = basic_tips + sport_tips

        print("Tips for Improvement:", tips)

        visualize_comparison(args.teacher, args.student, output_path=args.output, threshold=args.threshold)
        accuracy_scores = compute_accuracy_score(differences, threshold=args.threshold)
        avg_accuracy = np.mean(accuracy_scores)
        print(f"Overall Pose Accuracy: {avg_accuracy:.2f}%")
        export_differences_to_csv(differences, output_csv="session_differences.csv")
        plot_average_errors(avg_diffs, output_path="session_avg_errors.png")

        # Define which joints you want to track over time.
        # For example, here we use RIGHT_WRIST, LEFT_KNEE, and RIGHT_ANKLE:
        key_joints = [
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        plot_joint_errors_over_time(differences, key_joints, output_prefix="session_joint_errors")

if __name__ == "__main__":
    main()