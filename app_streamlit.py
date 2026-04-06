from pathlib import Path

import streamlit as st

from coachai_backend.core import run_offline_session

st.set_page_config(page_title="CoachAI", layout="wide")
st.title("CoachAI Pose Comparison")

with st.sidebar:
    st.header("Session Settings")
    threshold = st.slider("Threshold", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    sport = st.selectbox("Sport", ["general", "fencing", "skating"])

teacher_file = st.file_uploader("Teacher video (.mp4)", type=["mp4"], key="teacher")
student_file = st.file_uploader("Student video (.mp4)", type=["mp4"], key="student")

workspace = Path("streamlit_runs")
workspace.mkdir(parents=True, exist_ok=True)


def _save_upload(upload, suffix):
    target = workspace / f"{suffix}_{upload.name}"
    target.write_bytes(upload.getbuffer())
    return str(target)


if st.button("Run comparison", type="primary"):
    if not teacher_file or not student_file:
        st.error("Upload both teacher and student videos.")
    else:
        teacher_path = _save_upload(teacher_file, "teacher")
        student_path = _save_upload(student_file, "student")

        with st.spinner("Processing videos..."):
            result = run_offline_session(
                teacher_video=teacher_path,
                student_video=student_path,
                threshold=threshold,
                sport=sport,
                normal_output=str(workspace / "output_comparison_normal.mp4"),
                dynamic_output=str(workspace / "output_comparison_dynamic.mp4"),
                csv_output=str(workspace / "session_differences.csv"),
                avg_error_plot=str(workspace / "session_avg_errors.png"),
                joint_plot_prefix=str(workspace / "session_joint_errors"),
            )

        st.success("Comparison completed.")
        st.metric("Average accuracy", f"{result['average_accuracy']:.2f}%")
        st.write("Tips", result["tips"])

        st.subheader("Generated videos")
        normal_bytes = Path(result["normal_output"]).read_bytes()
        dynamic_bytes = Path(result["dynamic_output"]).read_bytes()
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Normal comparison")
            st.video(normal_bytes)
        with c2:
            st.caption("Dynamic comparison")
            st.video(dynamic_bytes)

        st.subheader("Charts")
        st.image(str(result["avg_error_plot"]), caption="Average error per joint")

        joint_plots = sorted(workspace.glob("session_joint_errors_*.png"))
        for chart in joint_plots:
            st.image(str(chart), caption=chart.name)

        st.subheader("Frame-by-frame differences")
        st.download_button(
            "Download CSV",
            data=Path(result["csv_output"]).read_bytes(),
            file_name="session_differences.csv",
            mime="text/csv",
        )
