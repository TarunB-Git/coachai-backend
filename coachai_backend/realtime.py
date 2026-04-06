import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
JOINT_NAMES = [lm.name for lm in mp_pose.PoseLandmark]


def calculate_joint_angle(p1, p2, p3):
    p1 = np.array(p1[:2])
    p2 = np.array(p2[:2])
    p3 = np.array(p3[:2])
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.degrees(np.arccos(cos_theta))


def estimate_center_of_gravity(keypoints):
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
    return {
        "arms": [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value,
        ],
        "legs": [
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        ],
        "torso": [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
        ],
    }


def calculate_difference(kp1, kp2):
    diffs = []
    for a, b in zip(kp1, kp2):
        if a and b:
            diffs.append(np.linalg.norm(np.array(a) - np.array(b)))
        else:
            diffs.append(None)
    return diffs


def generate_heuristics(avg_diffs, differences, sport="general", joint_threshold=0.1, teacher_kps=None, student_kps=None):
    tips = []

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

    region_scores = {"arms": 0, "legs": 0, "torso": 0}
    region_joints = get_body_region_joints()
    for region, joints in region_joints.items():
        region_diffs = [avg_diffs[i] * joint_weights[i] for i in joints if not np.isnan(avg_diffs[i])]
        region_scores[region] = np.mean(region_diffs) if region_diffs else 0

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
                    if angle_diff > 10:
                        key = f"{side}_ELBOW"
                        if key not in angle_tips or angle_diff > angle_tips[key][0]:
                            angle_tips[key] = (angle_diff, f"Adjust {side.lower()} elbow angle (off by {angle_diff:.1f}°).")

    if sport.lower() == "fencing":
        knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
        elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        if avg_diffs[knee_idx] > joint_threshold and avg_diffs[elbow_idx] > joint_threshold:
            tips.append("Lead knee and rear arm misaligned; check balance.")
        if (
            avg_diffs[mp_pose.PoseLandmark.RIGHT_WRIST.value] > joint_threshold
            and avg_diffs[mp_pose.PoseLandmark.RIGHT_ELBOW.value] > joint_threshold
        ):
            tips.append("Right wrist and elbow off; maintain guard position.")
    elif sport.lower() == "skating":
        if (
            avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > joint_threshold
            and avg_diffs[mp_pose.PoseLandmark.RIGHT_KNEE.value] > joint_threshold
        ):
            tips.append("Unstable knees; practice low stance.")
    else:
        if (
            avg_diffs[mp_pose.PoseLandmark.LEFT_SHOULDER.value] > joint_threshold
            and avg_diffs[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] > joint_threshold
        ):
            tips.append("Shoulders raised too high; relax posture.")

    if len(differences) > 10:
        sudden_deviations = []
        for i in range(1, len(differences)):
            if differences[i] and differences[i - 1]:
                frame_diff = np.nanmean(
                    [
                        abs(differences[i][j] - differences[i - 1][j])
                        for j in range(len(differences[i]))
                        if differences[i][j] is not None and differences[i - 1][j] is not None
                    ]
                )
                if frame_diff > joint_threshold * 1.5:
                    sudden_deviations.append(i)
        if sudden_deviations:
            tips.append(f"Sudden deviations at frames {sudden_deviations}; ensure smooth motion.")

    if sport.lower() == "fencing":
        if avg_diffs[mp_pose.PoseLandmark.LEFT_KNEE.value] > joint_threshold:
            tips.append("Align lead leg (left knee) for lunges.")
        if avg_diffs[mp_pose.PoseLandmark.RIGHT_ELBOW.value] > joint_threshold:
            tips.append("Steady rear arm (right elbow) for balance.")

    if teacher_kps and student_kps:
        t_cog = estimate_center_of_gravity(teacher_kps[0] if teacher_kps else None)
        s_cog = estimate_center_of_gravity(student_kps[0] if student_kps else None)
        if t_cog and s_cog:
            cog_diff = np.hypot(t_cog[0] - s_cog[0], t_cog[1] - s_cog[1])
            if cog_diff > 0.05:
                tips.append(f"Center of gravity off by {cog_diff:.3f}; adjust stance.")

    for i, joint in enumerate(JOINT_NAMES):
        error_frames = sum(1 for frame in differences if frame[i] is not None and frame[i] > joint_threshold)
        if error_frames / len(differences) > 0.65:
            tips.append(f"{joint} misaligned in {error_frames / len(differences) * 100:.1f}% of frames.")

    tips.extend([tip for _, tip in sorted(angle_tips.values(), key=lambda x: x[0], reverse=True)[:2]])

    if not tips:
        weighted_avg_diff = np.mean([avg_diffs[i] * joint_weights[i] for i in range(len(avg_diffs))])
        if weighted_avg_diff < joint_threshold * 0.5:
            tips.append("Excellent movement consistency!")
        elif weighted_avg_diff < joint_threshold:
            tips.append("Good performance; refine specific areas.")
        else:
            tips.append("High error; review technique.")

    for region, score in region_scores.items():
        if score > joint_threshold:
            tips.append(f"High error in {region} (score: {score:.3f}); focus here.")

    tip_scores = []
    for tip in tips:
        score = 0
        for i, joint in enumerate(JOINT_NAMES):
            if joint.lower() in tip.lower() and not np.isnan(avg_diffs[i]):
                score = max(score, avg_diffs[i])
        tip_scores.append((score, tip))

    tip_scores.sort(reverse=True)
    return [tip for _, tip in tip_scores[:2]], region_scores


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
    if not teacher_kps:
        print("Error: No valid teacher keypoints provided.")
        return

    cv2.namedWindow("Live Pose Feedback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Pose Feedback", 1280, 720)
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    frame_count = 0
    differences = []
    tips = []
    last_update = 0

    teacher_pose = landmark_pb2.NormalizedLandmarkList()
    for kp in teacher_kps:
        landmark = teacher_pose.landmark.add()
        landmark.x, landmark.y, landmark.z = kp

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: No frame received from webcam.")
            break

        frame = cv2.resize(frame, (1280, 720))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        if res.pose_landmarks and teacher_kps:
            student_kps = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            diffs = calculate_difference(teacher_kps, student_kps)
            differences.append(diffs)

            if len(differences) > 30:
                differences.pop(0)

            if frame_count % 30 == 0:
                avg_diffs = np.nanmean(differences, axis=0)
                if isinstance(avg_diffs, np.ndarray):
                    new_tips, _ = generate_heuristics(
                        avg_diffs,
                        differences,
                        sport=sport,
                        joint_threshold=threshold,
                        teacher_kps=[teacher_kps],
                        student_kps=[student_kps],
                    )
                    tips = new_tips[:2]
                    last_update = frame_count

            valid_diffs = [d for d in diffs if d is not None]
            mean_diff = np.mean(valid_diffs) if valid_diffs else 0
            accuracy = max(0, 100 - (mean_diff * 100))

            teacher_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2)
            overlay = frame.copy()
            mp_drawing.draw_landmarks(overlay, teacher_pose, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=teacher_style)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            overlay = frame.copy()
            for i, d in enumerate(diffs):
                if d is not None and d > threshold:
                    h, w = frame.shape[:2]
                    lm = res.pose_landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    pulse = 0.5 * (1 + np.sin(2 * np.pi * frame_count / 20))
                    radius = int(5 + 3 * pulse)
                    color = (0, 0, 255) if d > threshold * 1.5 else (0, 255, 255)
                    cv2.circle(overlay, (x, y), radius, color, -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(frame, (5, 5, 220, 45), (0, 0, 0), -1)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.rectangle(frame, (900, 50, 1270, 300), (0, 0, 0), -1)
            cv2.putText(frame, "Feedback", (910, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            for i, tip in enumerate(tips):
                cv2.putText(frame, tip, (910, 110 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, f"Last updated: frame {last_update}", (910, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (5, 5, 200, 40), (255, 255, 255), -1)
            cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Live Pose Feedback", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
