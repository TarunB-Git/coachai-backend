import cv2
import mediapipe as mp
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def calculate_joint_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) at p2 formed by points p1-p2-p3.
    p1, p2, p3 are tuples (x, y, z) or (x, y).
    """
    p1 = np.array(p1[:2])  # Use only x, y for 2D angle
    p2 = np.array(p2[:2])
    p3 = np.array(p3[:2])
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def estimate_center_of_gravity(keypoints):
    """
    Estimate center of gravity based on hip and foot positions.
    Returns (x, y) coordinates normalized to [0,1].
    """
    if not keypoints:
        return None
    hip_left = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip_right = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
    foot_left = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    foot_right = keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    if all([hip_left, hip_right, foot_left, foot_right]):
        x = (hip_left[0] + hip_right[0] + foot_left[0] + foot_right[0]) / 4
        y = (hip_left[1] + hip_right[1] + foot_left[1] + foot_right[1]) / 4
        return (x, y)
    return None

def get_body_region_joints():
    """
    Define joint indices for body regions: arms, legs, torso.
    Returns a dictionary mapping region names to lists of joint indices.
    """
    return {
        "arms": [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value
        ],
        "legs": [
            mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ],
        "torso": [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value
        ]
    }

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
JOINT_NAMES = [lm.name for lm in mp_pose.PoseLandmark]

def generate_heuristics(avg_diffs, differences, sport="general", joint_threshold=0.1, teacher_kps=None, student_kps=None):
    """
    Generates a list of recommendations, prioritizing specific tips and avoiding contradictions.
    
    Parameters:
      avg_diffs: 1D numpy array with average differences for each joint.
      differences: list of lists (frames x joints) with differences (for variance and timing).
      sport: a string indicating the sport context ('fencing', 'skating', 'general').
      joint_threshold: baseline threshold for triggering a tip.
      teacher_kps: list of teacher keypoints per frame (for angle calculations).
      student_kps: list of student keypoints per frame (for angle calculations).
      
    Returns:
      A list of heuristic tips (strings) and a dictionary of per-region scores.
    """
    tips = []
    variances = np.std(differences, axis=0) if differences and differences[0] is not None else np.zeros_like(avg_diffs)
    
    # Joint weights
    joint_weights = {i: 1.0 for i in range(len(JOINT_NAMES))}
    if sport.lower() == "fencing":
        joint_weights[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 1.5
        joint_weights[mp_pose.PoseLandmark.RIGHT_ELBOW.value] = 1.3
        joint_weights[mp_pose.PoseLandmark.LEFT_KNEE.value] = 1.2
    elif sport.lower() == "skating":
        joint_weights[mp_pose.PoseLandmark.LEFT_KNEE.value] = 1.5
        joint_weights[mp_pose.PoseLandmark.RIGHT_KNEE.value] = 1.5
        joint_weights[mp_pose.PoseLandmark.LEFT_ANKLE.value] = 1.3
        joint_weights[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = 1.3
    
    # Per-region scoring
    region_scores = {"arms": 0, "legs": 0, "torso": 0}
    region_joints = get_body_region_joints()
    for region, joints in region_joints.items():
        region_diffs = [avg_diffs[i] * joint_weights[i] for i in joints if not np.isnan(avg_diffs[i])]
        region_scores[region] = np.mean(region_diffs) if region_diffs else 0
    
    # Angle-based heuristics (limit to one per joint)
    angle_tips = {}
    if teacher_kps and student_kps:
        for t_kp, s_kp in zip(teacher_kps, student_kps):
            if t_kp and s_kp:
                for side in ["LEFT", "RIGHT"]:
                    shoulder = mp_pose.PoseLandmark[f"{side}_SHOULDER"].value
                    elbow = mp_pose.PoseLandmark[f"{side}_ELBOW"].value
                    wrist = mp_pose.PoseLandmark[f"{side}_WRIST"].value
                    t_angle = calculate_joint_angle(t_kp[shoulder], t_kp[elbow], t_kp[wrist])
                    s_angle = calculate_joint_angle(s_kp[shoulder], s_kp[elbow], s_kp[wrist])
                    angle_diff = abs(t_angle - s_angle)
                    if angle_diff > 10:  # ±10° tolerance
                        key = f"{side}_ELBOW"
                        if key not in angle_tips or angle_diff > angle_tips[key][0]:
                            angle_tips[key] = (angle_diff, f"{side} elbow angle off by {angle_diff:.1f}°. Adjust to match reference.")
    
    # Multi-joint combination heuristics
    if sport.lower() == "fencing":
        knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
        elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        if avg_diffs[knee_idx] > joint_threshold and avg_diffs[elbow_idx] > joint_threshold:
            tips.append("Lead knee too straight and rear arm misaligned; you may be off-balance.")
        if (avg_diffs[mp_pose.PoseLandmark.RIGHT_WRIST.value] > joint_threshold and
            avg_diffs[mp_pose.PoseLandmark.RIGHT_ELBOW.value] > joint_threshold):
            tips.append("Both right wrist and elbow are off; focus on maintaining guard position.")
    
    elif sport.lower() == "skating":
        if (avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > joint_threshold and
            avg_diffs[mp_pose.PoseLandmark.RIGHT_KNEE.value] > joint_threshold):
            tips.append("Both knees are unstable; practice low stance for better balance.")
    
    else:
        if (avg_diffs[mp_pose.PoseLandmark.LEFT_SHOULDER.value] > joint_threshold and
            avg_diffs[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] > joint_threshold):
            tips.append("Both shoulders are raised too high; relax for better posture.")
    
    # Timing heuristics
    if len(differences) > 10:
        sudden_deviations = []
        for i in range(1, len(differences)):
            if differences[i] and differences[i-1]:
                frame_diff = np.nanmean([abs(differences[i][j] - differences[i-1][j])
                                         for j in range(len(differences[i]))
                                         if differences[i][j] is not None and differences[i-1][j] is not None])
                if frame_diff > joint_threshold * 1.5:
                    sudden_deviations.append(i)
        if sudden_deviations:
            tips.append(f"Sudden deviations detected at frames {sudden_deviations}. Ensure smooth follow-through.")
    
    # Side-dependence for fencing
    if sport.lower() == "fencing":
        if avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > joint_threshold:
            tips.append("Lead leg (left knee) needs better alignment for lunges.")
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_ELBOW.value] > joint_threshold:
            tips.append("Rear arm (right elbow) is misaligned; keep it steady for balance.")
    
    # Balance assessment
    if teacher_kps and student_kps:
        t_cog = estimate_center_of_gravity(teacher_kps[0] if teacher_kps else None)
        s_cog = estimate_center_of_gravity(student_kps[0] if student_kps else None)
        if t_cog and s_cog:
            cog_diff = np.hypot(t_cog[0] - s_cog[0], t_cog[1] - s_cog[1])
            if cog_diff > 0.05:
                tips.append(f"Center of gravity off by {cog_diff:.3f}. Adjust stance to avoid leaning.")
    
    # Tracking over time
    consistent_issues = []
    for i, joint in enumerate(JOINT_NAMES):
        error_frames = sum(1 for frame in differences if frame[i] is not None and frame[i] > joint_threshold)
        if error_frames / len(differences) > 0.65:
            consistent_issues.append(f"{joint} misaligned in {error_frames/len(differences)*100:.1f}% of frames.")
    tips.extend(consistent_issues)
    
    # Add angle-based tips (limited to top 3)
    angle_tips_sorted = sorted(angle_tips.values(), key=lambda x: x[0], reverse=True)[:3]
    tips.extend([tip for _, tip in angle_tips_sorted])
    
    # General performance summary (only if no specific tips)
    if not tips:  # Only add generic tip if no specific issues found
        weighted_avg_diff = np.mean([avg_diffs[i] * joint_weights[i] for i in range(len(avg_diffs))])
        if weighted_avg_diff < joint_threshold * 0.5:
            tips.append("Excellent overall movement consistency!")
        elif weighted_avg_diff < joint_threshold:
            tips.append("Good performance, but refine specific areas.")
        else:
            tips.append("High overall error; review technique thoroughly.")
    
    # Region-based feedback
    for region, score in region_scores.items():
        if score > joint_threshold:
            tips.append(f"{region.capitalize()} region shows high error (score: {score:.3f}). Focus on this area.")
    
    return tips, region_scores
    

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


def visualize_comparison_normal(teacher_video, student_video, output_path="output_comparison_normal.mp4", threshold=0.1):
    cap1 = cv2.VideoCapture(teacher_video)
    cap2 = cv2.VideoCapture(student_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (720, 480))
    
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
        
        if res1.pose_landmarks and res2.pose_landmarks:
            kp1 = [(lm.x, lm.y, lm.z) for lm in res1.pose_landmarks.landmark]
            kp2 = [(lm.x, lm.y, lm.z) for lm in res2.pose_landmarks.landmark]
            diffs = calculate_difference(kp1, kp2)
            
            valid_diffs = [(i, d) for i, d in enumerate(diffs) if d is not None]
            valid_diffs.sort(key=lambda x: x[1], reverse=True)
            top_errors = valid_diffs[:1]
            
            teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
            mp_drawing.draw_landmarks(frame2, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=teacher_style)
            
            for i, d in enumerate(diffs):
                h, w = frame2.shape[:2]
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
                    cv2.circle(frame2, (x, y), 5, color, -1)
                    if label:
                        cv2.putText(frame2, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.circle(frame2, (x, y), 5, (0, 255, 0), -1)
            
            overlay = frame2.copy()
            for i, d in top_errors:
                if d > threshold:
                    x, y = int(res2.pose_landmarks.landmark[i].x * w), int(res2.pose_landmarks.landmark[i].y * h)
                    intensity = min(255, int(d * 1000))
                    cv2.circle(overlay, (x, y), 20, (0, 0, intensity), -1)
            alpha = 0.3
            frame2 = cv2.addWeighted(overlay, alpha, frame2, 1 - alpha, 0)
            
            mp_drawing.draw_landmarks(frame1, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame2, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        combined = cv2.hconcat([frame1, frame2])
        
        if res1.pose_landmarks and res2.pose_landmarks:
            valid_diffs = [d for d in diffs if d is not None]
            avg_diff = np.mean(valid_diffs) if valid_diffs else 0
            if avg_diff > 1.5 * threshold:
                cv2.putText(combined, "High Error Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                for _ in range(3):
                    out.write(combined)
            else:
                out.write(combined)
        else:
            out.write(combined)
    
    cap1.release()
    cap2.release()
    out.release()
    print(f"Normal comparison video saved to {output_path}")

def visualize_comparison_dynamic(teacher_video, student_video, output_path="output_comparison_dynamic.mp4", threshold=0.1):
    cap1 = cv2.VideoCapture(teacher_video)
    cap2 = cv2.VideoCapture(student_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (720, 480))
    
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
            
            if res1.pose_landmarks:
                teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
                mp_drawing.draw_landmarks(frame2, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=teacher_style)
            
            if res2.pose_landmarks:
                overlay = frame2.copy()
                for i, d in top_errors:
                    h, w = frame2.shape[:2]
                    p2 = res2.pose_landmarks.landmark[i]
                    x, y = int(p2.x * w), int(p2.y * h)
                    if d > threshold:
                        intensity = min(255, int(d * 1000))
                        cv2.circle(overlay, (x, y), 20, (0, 0, intensity), -1)
                alpha = 0.3
                frame2 = cv2.addWeighted(overlay, alpha, frame2, 1 - alpha, 0)
                mp_drawing.draw_landmarks(frame2, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            combined_base = cv2.hconcat([frame1, frame2])
            
            num_repeats = 20 if avg_diff <= 1.5 * threshold else 30
            for t in range(num_repeats):
                combined = combined_base.copy()
                
                pulse = 0.5 * (1 + np.sin(2 * np.pi * t / num_repeats))
                circle_radius = int(5 + 3 * pulse)
                text_scale = 0.7 + 0.2 * pulse
                
                if res2.pose_landmarks:
                    for i, d in top_errors:
                        h, w = frame2.shape[:2]
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
                        cv2.circle(combined, (x + 360, y), circle_radius, color, -1)
                        if label:
                            cv2.putText(combined, label, (x + 370, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.6, color, 1)
                
                if avg_diff > 1.5 * threshold:
                    cv2.putText(combined, "High Error Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 2)
                
                if top_errors:
                    summary = "Focus on: " + ", ".join(JOINT_NAMES[i] for i, _ in top_errors[:3])
                    opacity = 1.0
                    if t < num_repeats // 2:
                        opacity = t / (num_repeats // 2)
                    else:
                        opacity = 1 - (t - num_repeats // 2) / (num_repeats // 2)
                    overlay = combined.copy()
                    cv2.putText(overlay, summary, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    cv2.addWeighted(overlay, opacity, combined, 1 - opacity, 0, combined)
                
                out.write(combined)
            
            window_diffs = []
            window_frames = []
    
    cap1.release()
    cap2.release()
    out.release()
    print(f"Dynamic comparison video saved to {output_path}")

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


def live_pose_feedback(teacher_kps, threshold=0.1, sport="general"):
    """
    Provides real-time feedback by comparing live webcam feed against teacher's reference keypoints.
    Displays visual alerts and textual tips for posture corrections.
    """
    cv2.namedWindow('Live Pose Feedback', cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam. Check device connection.")
        return
    
    frame_count = 0
    differences = []
    tips = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: No frame received from webcam.")
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        
        if res.pose_landmarks and teacher_kps:
            student_kps = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            # Compute differences
            diffs = calculate_difference(teacher_kps, student_kps)
            differences.append(diffs)
            
            # Keep only the last 30 frames to avoid excessive memory usage
            if len(differences) > 30:
                differences.pop(0)
            
            # Generate tips every 30 frames
            if frame_count % 30 == 0:
                avg_diffs = np.nanmean(differences, axis=0)
                if isinstance(avg_diffs, np.ndarray):
                    new_tips, _ = generate_heuristics(
                        avg_diffs,
                        differences,
                        sport=sport,
                        joint_threshold=threshold,
                        teacher_kps=[teacher_kps],
                        student_kps=[student_kps]
                    )
                    tips = new_tips[:3]  # Limit to top 3 tips
            
            # Calculate accuracy
            valid_diffs = [d for d in diffs if d is not None]
            mean_diff = np.mean(valid_diffs) if valid_diffs else 0
            accuracy = max(0, 100 - (mean_diff * 100))
            
            # Visual alerts
            overlay = frame.copy()
            for i, d in enumerate(diffs):
                if d is not None and d > threshold:
                    h, w = frame.shape[:2]
                    lm = res.pose_landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    # Pulsing red circle for high error
                    pulse = 0.5 * (1 + np.sin(2 * np.pi * frame_count / 20))
                    radius = int(5 + 3 * pulse)
                    color = (0, 0, 255) if d > threshold * 1.5 else (0, 255, 255)
                    cv2.circle(overlay, (x, y), radius, color, -1)
                    # Label for high-error joints
                    if d > threshold * 1.5:
                        cv2.putText(overlay, f"{JOINT_NAMES[i]}: High", (x + 10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Apply overlay with transparency
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display accuracy and tips
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i, tip in enumerate(tips):
                cv2.putText(frame, tip, (10, 60 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        else:
            # No landmarks detected
            cv2.putText(frame, "No pose detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Live Pose Feedback', frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
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
        avg_diffs, tips = summarize_differences(differences, teacher_kps=teacher_kps, student_kps=student_kps,
                                               sport=args.sport, threshold=args.threshold)
        print("Average Differences:", avg_diffs)
        print("Tips for Improvement:", tips)
        
        # Generate both comparison videos
        visualize_comparison_normal(args.teacher, args.student, output_path="output_comparison_normal.mp4", 
                                  threshold=args.threshold)
        visualize_comparison_dynamic(args.teacher, args.student, output_path="output_comparison_dynamic.mp4", 
                                   threshold=args.threshold)
        
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