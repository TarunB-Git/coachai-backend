CoachAI

CoachAI is a Python application for comparing human poses between a teacher (coach) video and a student (trainee) video, providing both offline and real-time feedback, analytics, and visualizations.

🚀 Features

Offline Mode:

Extracts pose keypoints from teacher and student videos using MediaPipe.

Computes per-frame and average joint differences.

Generates improvement tips based on heuristic thresholds for specific joints.

Creates a side-by-side comparison video highlighting deviating joints.

Exports a CSV of frame-by-frame differences.

Plots average error per joint and error-over-time charts for selected joints.

Real-Time Mode:

Captures live webcam feed of student.

Compares to a reference pose from the teacher video.

Displays live pose overlay with accuracy score and highlighted joints.

📦 Installation

Clone this repository:

git clone https://github.com/TarunB-Git/coachai-backend 
cd coachai-backend

Create and activate a virtual environment (optional but recommended):

python3 -m venv venv source venv/bin/activate

Install dependencies:

pip install opencv-python mediapipe numpy matplotlib

⚙️ Usage

The script coachai.py supports both offline and live modes via command-line arguments.

Offline Mode

Compare two videos and produce analytics and visual feedback:

python coachai.py --teacher path/to/teacher.mp4 --student path/to/student.mp4 --threshold 0.1 --output comparison.mp4

--teacher : Path to the coach/teacher video.

--student : Path to the student/practice video.

--threshold : Difference threshold (normalized) for highlighting joints (default: 0.1).

--output : Path for the side-by-side comparison video (default: output.mp4).

Outputs generated:

comparison.mp4 : Side-by-side video with skeletons and red highlights.

session_differences.csv : Frame-by-frame joint error values.

session_avg_errors.png : Bar chart of average error per joint.

session_joint_errors_.png : Error-over-time plots for selected joints.

Live Mode

Provide real-time feedback via webcam:

python coachai.py --teacher path/to/teacher.mp4 --live --threshold 0.1

--teacher : Path to the coach/teacher video (used to extract reference pose).

--live : Flag to enable webcam mode.

--threshold : Difference threshold for highlighting (default: 0.1).

A window named Live Pose Feedback will appear, showing:

Real-time skeleton overlay.

Live accuracy percentage.

Red dots on joints exceeding the threshold.

Press Esc to exit.

🔧 Configuration

Adjust threshold to control sensitivity.

Modify joint heuristics in summarize_differences() for sport-specific advice.

Extend analytics functions to include additional joints or output formats.

🛠️ Development

Code structure is organized into:

Offline processing: keypoint extraction, comparison, summarization, visualization.

Real-time: reference loading and live feedback loop.

Analytics: CSV export and plotting.

Feel free to open issues or contribute enhancements.

📜 License

This project is released under the MIT License. See LICENSE for details.

🙏 Acknowledgements

MediaPipe for real-time pose estimation.

OpenCV for video processing.

Matplotlib for analytics plotting.

Enjoy coaching and training with AI-powered insights!