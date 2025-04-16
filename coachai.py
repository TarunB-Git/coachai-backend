import cv2
import mediapipe as mp
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from live_realtime import load_teacher_reference, live_pose_feedback, calculate_joint_angle, estimate_center_of_gravity, get_body_region_joints, generate_heuristics

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
JOINT_NAMES = [lm.name for lm in mp_pose.PoseLandmark]

def calculate_difference(kp1, kp2):
    diffs = []
    for a, b in zip(kp1, kp2):
        if a and b:
            diffs.append(np.linalg.norm(np.array(a) - np.array(b)))
        else:
            diffs.append(None)
    return diffs

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

def compare_videos(teacher_kps, student_kps):
    frame_differences = []
    for t_kp, s_kp in zip(teacher_kps, student_kps):
        if t_kp and s_kp:
            frame_differences.append(calculate_difference(t_kp, s_kp))
    return frame_differences

def summarize_differences(differences, teacher_kps=None, student_kps=None, sport="general", threshold=0.1):
    if not differences:
        print("No valid frame comparisons found.")
        return [], ["No movement data available for analysis. Check video quality or pose detection."]
    avg_diffs = np.nanmean(differences, axis=0)
    if not isinstance(avg_diffs, np.ndarray):
        print("Average differences is not an array. Something went wrong.")
        return [], ["Pose data may be incomplete."]
    
    # Generate tips with enhanced heuristics
    tips, region_scores = generate_heuristics(avg_diffs, differences, sport=sport,
                                           joint_threshold=threshold,
                                           teacher_kps=teacher_kps,
                                           student_kps=student_kps)
    
    # Print region scores
    print("Per-Region Error Scores:")
    for region, score in region_scores.items():
        print(f"  {region.capitalize()}: {score:.3f}")
    
    # Heuristic accuracy score
    valid_diffs = avg_diffs[~np.isnan(avg_diffs)]
    mean_diff = np.mean(valid_diffs)
    accuracy_score = max(0, 100 - (mean_diff * 100))
    print(f"Movement Accuracy Score: {accuracy_score:.2f}%")
    
    return avg_diffs, tips

def visualize_comparison(teacher_video, student_video, normal_output="output_comparison_normal.mp4", 
                        dynamic_output="output_comparison_dynamic.mp4", threshold=0.1):
    cap1 = cv2.VideoCapture(teacher_video)
    cap2 = cv2.VideoCapture(student_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_normal = cv2.VideoWriter(normal_output, fourcc, 20.0, (720, 480))
    out_dynamic = cv2.VideoWriter(dynamic_output, fourcc, 20.0, (720, 480))
    
    frame_count = 0
    window_size = 20
    window_diffs = []
    window_frames = []
    
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
        frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
        frame1 = cv2.resize(frame1, (360, 480))
        frame2 = cv2.resize(frame2, (360, 480))
        
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        res1 = pose.process(img1)
        res2 = pose.process(img2)
        
        diffs = None
        if res1.pose_landmarks and res2.pose_landmarks:
            kp1 = [(lm.x, lm.y, lm.z) for lm in res1.pose_landmarks.landmark]
            kp2 = [(lm.x, lm.y, lm.z) for lm in res2.pose_landmarks.landmark]
            diffs = calculate_difference(kp1, kp2)
        
        window_diffs.append(diffs)
        window_frames.append((frame1, frame2, res1, res2))
        frame_count += 1
        
        # Normal video processing
        frame_normal = frame2.copy()
        if res1.pose_landmarks and res2.pose_landmarks:
            teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
            mp_drawing.draw_landmarks(frame_normal, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=teacher_style)
            
            valid_diffs = [(i, d) for i, d in enumerate(diffs) if d is not None]
            valid_diffs.sort(key=lambda x: x[1], reverse=True)
            top_errors = valid_diffs[:1]
            
            for i, d in enumerate(diffs):
                h, w = frame_normal.shape[:2]
                p2 = res2.pose_landmarks.landmark[i]
                x, y = int(p2.x * w), int(p2.y * h)
                if (i, d) in top_errors and d is not None:
                    if d > threshold * 1.5:
                        color = (0, 0, 255)
                        label = f"{JOINT_NAMES[i]}: High"
                    elif d > threshold:
                        color = (0, 255, 255)
                        label = f"{JOINT_NAMES[i]}: Adjust"
                    else:
                        color = (0, 255, 0)
                        label = None
                    cv2.circle(frame_normal, (x, y), 5, color, -1)
                    if label:
                        cv2.putText(frame_normal, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.circle(frame_normal, (x, y), 5, (0, 255, 0), -1)
            
            overlay = frame_normal.copy()
            for i, d in top_errors:
                if d > threshold:
                    x, y = int(res2.pose_landmarks.landmark[i].x * w), int(res2.pose_landmarks.landmark[i].y * h)
                    intensity = min(255, int(d * 1000))
                    cv2.circle(overlay, (x, y), 20, (0, 0, intensity), -1)
            alpha = 0.3
            frame_normal = cv2.addWeighted(overlay, alpha, frame_normal, 1 - alpha, 0)
            
            mp_drawing.draw_landmarks(frame_normal, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        combined_normal = cv2.hconcat([frame1, frame_normal])
        if res1.pose_landmarks and res2.pose_landmarks:
            valid_diffs = [d for d in diffs if d is not None]
            avg_diff = np.mean(valid_diffs) if valid_diffs else 0
            if avg_diff > 1.5 * threshold:
                cv2.putText(combined_normal, "High Error Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                for _ in range(3):
                    out_normal.write(combined_normal)
            else:
                out_normal.write(combined_normal)
        else:
            out_normal.write(combined_normal)
        
        # Dynamic video processing
        if len(window_frames) == window_size or (not ret1 or not ret2):
            max_diffs = [None] * len(JOINT_NAMES)
            if any(d is not None for d in window_diffs):
                for i in range(len(JOINT_NAMES)):
                    valid_diffs = [d[i] for d in window_diffs if d is not None and d[i] is not None]
                    max_diffs[i] = max(valid_diffs) if valid_diffs else None
            
            valid_diffs = [(i, d) for i, d in enumerate(max_diffs) if d is not None]
            valid_diffs.sort(key=lambda x: x[1], reverse=True)
            top_errors = valid_diffs[:1]
            avg_diff = np.mean([d for _, d in top_errors]) if top_errors else 0
            
            mid_idx = min(len(window_frames) // 2, len(window_frames) - 1)
            frame1, frame2, res1, res2 = window_frames[mid_idx]
            frame_dynamic = frame2.copy()
            
            if res1.pose_landmarks:
                teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
                mp_drawing.draw_landmarks(frame_dynamic, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=teacher_style)
            
            if res2.pose_landmarks:
                overlay = frame_dynamic.copy()
                for i, d in top_errors:
                    h, w = frame_dynamic.shape[:2]
                    p2 = res2.pose_landmarks.landmark[i]
                    x, y = int(p2.x * w), int(p2.y * h)
                    if d > threshold:
                        intensity = min(255, int(d * 1000))
                        cv2.circle(overlay, (x, y), 20, (0, 0, intensity), -1)
                alpha = 0.3
                frame_dynamic = cv2.addWeighted(overlay, alpha, frame_dynamic, 1 - alpha, 0)
                mp_drawing.draw_landmarks(frame_dynamic, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            combined_base = cv2.hconcat([frame1, frame_dynamic])
            
            num_repeats = 20 if avg_diff <= 1.5 * threshold else 30
            for t in range(num_repeats):
                combined_dynamic = combined_base.copy()
                
                pulse = 0.5 * (1 + np.sin(2 * np.pi * t / num_repeats))
                circle_radius = int(5 + 3 * pulse)
                text_scale = 0.7 + 0.2 * pulse
                
                if res2.pose_landmarks:
                    for i, d in top_errors:
                        h, w = frame_dynamic.shape[:2]
                        p2 = res2.pose_landmarks.landmark[i]
                        x, y = int(p2.x * w), int(p2.y * h)
                        if d > threshold * 1.5:
                            color = (0, 0, 255)
                            label = f"{JOINT_NAMES[i]}: High"
                        elif d > threshold:
                            color = (0, 255, 255)
                            label = f"{JOINT_NAMES[i]}: Adjust"
                        else:
                            color = (0, 255, 0)
                            label = None
                        cv2.circle(combined_dynamic, (x + 360, y), circle_radius, color, -1)
                        if label:
                            cv2.putText(combined_dynamic, label, (x + 370, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, color, 1)
                
                if avg_diff > 1.5 * threshold:
                    cv2.putText(combined_dynamic, "High Error Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 2)
                
                if top_errors:
                    summary = "Focus on: " + ", ".join(JOINT_NAMES[i] for i, _ in top_errors[:3])
                    opacity = 1.0
                    if t < num_repeats // 2:
                        opacity = t / (num_repeats // 2)
                    else:
                        opacity = 1 - (t - num_repeats // 2) / (num_repeats // 2)
                    overlay = combined_dynamic.copy()
                    cv2.putText(overlay, summary, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    cv2.addWeighted(overlay, opacity, combined_dynamic, 1 - opacity, 0, combined_dynamic)
                
                out_dynamic.write(combined_dynamic)
            
            window_diffs = []
            window_frames = []
    
    cap1.release()
    cap2.release()
    out_normal.release()
    out_dynamic.release()
    print(f"Normal comparison video saved to {normal_output}")
    print(f"Dynamic comparison video saved to {dynamic_output}")

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

def export_differences_to_csv(differences, output_csv="differences.csv"):
    """
    Exports frame-by-frame differences to a CSV file.
    
    Each row represents one frame, with columns:
      frame, followed by a column for each joint (named according to JOINT_NAMES)
    """
    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + JOINT_NAMES)
            for i, frame_diff in enumerate(differences):
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

def main():
    parser = argparse.ArgumentParser(description='CoachAI: Pose Comparison and Real-Time Feedback')
    parser.add_argument('--teacher', required=True, help='Path to teacher video')
    parser.add_argument('--student', help='Path to student video (omit for live mode)')
    parser.add_argument('--live', action='store_true', help='Enable real-time webcam feedback')
    parser.add_argument('--threshold', type=float, default=0.1, help='Highlight threshold for differences (normalized)')
    parser.add_argument('--sport', default='general', help='Sport context: fencing, skating, or general')

    args = parser.parse_args()
    if args.live:
        teacher_kps = load_teacher_reference(args.teacher)
        if not teacher_kps:
            print("Failed to load teacher reference keypoints.")
            return
        live_pose_feedback(teacher_kps, threshold=args.threshold, sport=args.sport)
    else:
        if not args.student:
            print("Please provide a student video in non-live mode.")
            return
        teacher_kps = extract_keypoints(args.teacher)
        student_kps = extract_keypoints(args.student)
        print(f"Teacher frames with keypoints: {sum(kp is not None for kp in teacher_kps)} / {len(teacher_kps)}")
        print(f"Student frames with keypoints: {sum(kp is not None for kp in student_kps)} / {len(student_kps)}")
        differences = compare_videos(teacher_kps, student_kps)
        avg_diffs, tips = summarize_differences(differences, teacher_kps=teacher_kps, student_kps=student_kps,
                                               sport=args.sport, threshold=args.threshold)
        print("Average Differences:", avg_diffs)
        print("Tips for Improvement:", tips)
        
        # Generate both comparison videos
        visualize_comparison(args.teacher, args.student, threshold=args.threshold)
        
        accuracy_scores = compute_accuracy_score(differences, threshold=args.threshold)
        avg_accuracy = np.mean(accuracy_scores)
        print(f"Overall Pose Accuracy: {avg_accuracy:.2f}%")
        export_differences_to_csv(differences, output_csv="session_differences.csv")
        plot_average_errors(avg_diffs, output_path="session_avg_errors.png")
        
        key_joints = [
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]
        plot_joint_errors_over_time(differences, key_joints, output_prefix="session_joint_errors")

if __name__ == "__main__":
    main()