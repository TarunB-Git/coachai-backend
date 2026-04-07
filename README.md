# CoachAI

CoachAI compares pose alignment between a teacher video and a student video, then produces feedback, analytics, and rendered comparison outputs.

## What changed

- Repository restructured into a reusable package: `coachai_backend/`
- Deprecated/fragile fully-pinned dependency list replaced with a maintained minimal set
- Added a web frontend with Streamlit (`app_streamlit.py`) to upload videos and view model outputs
- CLI flow retained through `coachai.py`

## Repository structure

- `coachai.py` — command-line entrypoint
- `app_streamlit.py` — Streamlit frontend
- `coachai_backend/core.py` — offline processing pipeline
- `coachai_backend/realtime.py` — real-time pose feedback logic
- `coachai_backend/analytics.py` — CSV + plot generation

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI usage

### Offline comparison

```bash
python coachai.py --teacher path/to/teacher.mp4 --student path/to/student.mp4 --threshold 0.1 --sport general
```

Generated outputs include:

- `output_comparison_normal.mp4`
- `output_comparison_dynamic.mp4`
- `session_differences.csv`
- `session_avg_errors.png`
- `session_joint_errors_*.png`

### Live webcam feedback

```bash
python coachai.py --teacher path/to/teacher.mp4 --live --threshold 0.1 --sport general
```

## Streamlit frontend usage

```bash
streamlit run app_streamlit.py
```

Then choose a mode in the sidebar:

- **Offline comparison**: upload teacher/student videos, run comparison, and inspect:
  - Accuracy metric
  - Improvement tips
  - Generated videos
  - Error charts
  - CSV export
- **Live session**: upload teacher video and start a webcam session to get real-time posture feedback against the coach reference.

Press `ESC` in the OpenCV live window to end the live session.
