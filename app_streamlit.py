from pathlib import Path

import streamlit as st

from coachai_backend.core import run_offline_session
from coachai_backend.realtime import live_pose_feedback, load_teacher_reference

st.set_page_config(page_title="CoachAI", layout="wide")
st.title("CoachAI Pose Comparison")

with st.sidebar:
    st.header("Session Settings")
    session_mode = st.radio("Mode", ["Offline comparison", "Live session"], index=0)
    threshold = st.slider("Threshold", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    sport = st.selectbox("Sport", ["general", "fencing", "skating"])

teacher_file = st.file_uploader("Teacher video (.mp4)", type=["mp4"], key="teacher")
student_file = None
if session_mode == "Offline comparison":
    student_file = st.file_uploader("Student video (.mp4)", type=["mp4"], key="student")
else:
    st.info("Live session uses your webcam and compares your posture to the uploaded teacher video.")

workspace = Path("streamlit_runs")
workspace.mkdir(parents=True, exist_ok=True)


def _save_upload(upload, suffix):
    target = workspace / f"{suffix}_{upload.name}"
    target.write_bytes(upload.getbuffer())
    return str(target)


if session_mode == "Offline comparison":
    if st.button("Run comparison", type="primary"):
        if not teacher_file or not student_file:
            st.error("Upload both teacher and student videos.")
        else:
            run_id = str(int(st.session_state.get("_last_run_id", 0) + 1))
            st.session_state["_last_run_id"] = int(run_id)

            teacher_path = _save_upload(teacher_file, f"{run_id}_teacher")
            student_path = _save_upload(student_file, f"{run_id}_student")
            normal_output = workspace / f"{run_id}_output_comparison_normal.mp4"
            dynamic_output = workspace / f"{run_id}_output_comparison_dynamic.mp4"
            csv_output = workspace / f"{run_id}_session_differences.csv"
            avg_plot = workspace / f"{run_id}_session_avg_errors.png"
            joint_prefix = workspace / f"{run_id}_session_joint_errors"

            with st.spinner("Processing videos..."):
                result = run_offline_session(
                    teacher_video=teacher_path,
                    student_video=student_path,
                    threshold=threshold,
                    sport=sport,
                    normal_output=str(normal_output),
                    dynamic_output=str(dynamic_output),
                    csv_output=str(csv_output),
                    avg_error_plot=str(avg_plot),
                    joint_plot_prefix=str(joint_prefix),
                )

            st.success("Comparison completed.")
            st.metric("Average accuracy", f"{result['average_accuracy']:.2f}%")
            st.write("Tips", result["tips"])

            st.subheader("Generated videos")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Normal comparison")
                st.video(str(result["normal_output"]), format="video/mp4")
            with c2:
                st.caption("Dynamic comparison")
                st.video(str(result["dynamic_output"]), format="video/mp4")

            st.subheader("Charts")
            st.image(str(result["avg_error_plot"]), caption="Average error per joint")

            joint_plots = sorted(workspace.glob(f"{joint_prefix.name}_*.png"))
            for chart in joint_plots:
                st.image(str(chart), caption=chart.name)

            st.subheader("Frame-by-frame differences")
            st.download_button(
                "Download CSV",
                data=Path(result["csv_output"]).read_bytes(),
                file_name="session_differences.csv",
                mime="text/csv",
            )
else:
    if st.button("Start live session", type="primary"):
        if not teacher_file:
            st.error("Upload a teacher video to start a live session.")
        else:
            teacher_path = _save_upload(teacher_file, "live_teacher")
            teacher_kps = load_teacher_reference(teacher_path)
            if not teacher_kps:
                st.error("Could not detect teacher pose reference from the uploaded video.")
            else:
                st.warning("Launching webcam session. Press ESC in the OpenCV window to end.")
                live_pose_feedback(teacher_kps, threshold=threshold, sport=sport)
