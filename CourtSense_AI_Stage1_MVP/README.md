# CourtSense AI — Stage I Working Prototype

A functional basketball coaching **proof of concept** combining actual video pose inference, human-reviewed tactical tagging, explainable lineup ranking, and workload discussion prompts.

## Quick start

Use **Python 3.10–3.12** (recommended 3.11). On a machine with internet access:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

First inference downloads `yolo11n-pose.pt` automatically (Ultralytics model weights). Subsequent runs can use cached weights offline. For a fully offline demo, download the weights ahead of time and keep the `.pt` file in the working directory. The weights are not bundled in this repository. Check Ultralytics model licensing before commercial use.

## What is actually automated?

- Upload an MP4/MOV/AVI and run a **real pretrained YOLO pose model** on sampled frames.
- Detect visible people and 17 COCO keypoints, draw pose skeletons and export annotated video.
- Measure 2D knee, hip and elbow angles and export time-stamped CSV.
- Display non-medical technique review prompts derived from measured angles.
- Calculate screen-space spread of detected people (a camera-dependent proxy, **not tactical court positioning**).

## What is assisted or simulated?

- Opponent tactics: **manually verified event tags** are summarized. The model does NOT recognize pick-and-roll, shooting outcomes or specific athletes automatically.
- Lineup: **transparent weighted scoring heuristic** over editable example statistics. Not a trained optimization model.
- Workload: manually entered RPE and minutes; no fatigue estimation from video and no injury probability.
- Tracking: first detected person per frame may change identity. This is not player re-identification.

## Suggested 3-minute Stage I video

1. (0:00–0:25) Problem: costly manual basketball video review.
2. (0:25–1:30) Upload a consented practice clip; run inference; show overlaid skeleton and measured CSV.
3. (1:30–2:10) Add a few opponent events and show tendency counts.
4. (2:10–2:40) Edit roster stats and compare five-player lineup scores.
5. (2:40–3:00) Limitations and roadmap: court calibration, jersey recognition, ball tracking, tactical event classification, longitudinal validation.

## Architecture

Streamlit UI → OpenCV frame sampler → Ultralytics YOLO pose → `analytics.py` measurements → CSV + annotated MP4.

Manual event tags → descriptive event counts. Player table → `lineup.py` heuristic rankings.

## Test

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py video_ai.py analytics.py lineup.py
```

## Known limitations and safeguards

- Camera movement, perspective, occlusion and multiple athletes may cause noisy or misleading pose measurements.
- No automatic injury diagnosis, clinical injury-risk prediction, or validated fatigue assessment.
- No automated player identities, game events, or strategic play recognition yet.
- Get informed consent for footage; avoid sharing identifiable athlete data publicly.
- Stage I prototype, not a production medical or scouting tool.
