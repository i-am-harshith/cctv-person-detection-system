# AI-Powered CCTV Person Detection and Crowd Monitoring System

A Streamlit-based computer vision project that detects people in CCTV videos, measures timestamp-wise crowd count, flags crowd alerts, and exports analytics reports.

## Problem Statement

CCTV footage often needs manual review to understand crowd density at different times. This project automates person detection on sampled video frames and converts the results into simple crowd-monitoring analytics suitable for security, facility management, and learning demonstrations.

## Project Overview

This project upgrades an older TensorFlow SSD MobileNet frozen-graph script into a cleaner ML portfolio application. The core objective remains the same: detect people in CCTV footage and report person count by timestamp. The upgraded version adds a Streamlit UI, configurable detection settings, alert logic, CSV/JSON reports, and a maintainable project structure.

## Features

- Upload CCTV video files through a Streamlit interface
- Preview the uploaded video before processing
- Configure confidence threshold for detections
- Analyze frames every 1, 2, or 5 seconds
- Set a crowd alert threshold
- Count only detections classified as `person`
- Display summary metrics for crowd monitoring
- Show timestamp-wise detection results in a table
- Highlight alert events
- Export CSV and JSON reports
- Optionally preview an annotated output video

## Tech Stack

- Python
- TensorFlow frozen graph inference with TensorFlow 2 compatibility wrappers
- OpenCV for video reading, frame processing, and annotations
- Streamlit for the web UI
- Pandas for tabular analytics
- NumPy for frame and detection array handling

## Architecture / Workflow

```text
Video Upload
    |
Save uploaded video locally
    |
Open video with OpenCV
    |
Sample frames every N seconds
    |
Run TensorFlow SSD MobileNet inference
    |
Filter detections by confidence and person class
    |
Calculate person count and crowd alert status
    |
Display Streamlit metrics, table, alerts, and reports
```

## Project Structure

```text
cctv-person-detection-system/
|
├── app.py
├── detection.py
├── utils/
│   ├── video_utils.py
│   ├── report_utils.py
│   ├── label_map_util.py
│   ├── people_class_util.py
│   └── visualization_utils.py
|
├── models/
│   └── README.md
|
├── sample_videos/
│   └── README.md
|
├── outputs/
│   └── .gitkeep
|
├── images/
│   └── README.md
|
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If TensorFlow protobuf compatibility issues appear, run:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Model Setup

Large model files are intentionally not committed to GitHub.

Place the SSD MobileNet frozen graph at:

```text
models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
```

If the model is stored elsewhere, set:

```bash
export CCTV_MODEL_PATH="/absolute/path/to/frozen_inference_graph.pb"
```

## How To Run

Start the Streamlit app:

```bash
streamlit run app.py
```

Then:

1. Upload a CCTV video.
2. Select confidence threshold, frame interval, and alert threshold.
3. Click `Run Detection`.
4. Review summary metrics, timestamp-wise results, alert rows, and exported reports.

## How Detection Works

The project uses the original TensorFlow Object Detection frozen graph approach with TensorFlow 2 compatibility wrappers:

- `tf.compat.v1.disable_eager_execution()`
- `tf.compat.v1.GraphDef()`
- `tf.compat.v1.import_graph_def()`
- `tf.compat.v1.Session()`
- `tf.io.gfile.GFile()`

OpenCV reads the video frame by frame. Instead of running inference on every frame, the app samples frames at the selected interval, such as every 2 seconds. Each sampled frame is passed to the TensorFlow graph. Detections are counted only when:

- the predicted class is `person`
- the confidence score is greater than or equal to the selected threshold

Alert status is generated as:

```text
Alert = Yes if person_count >= alert_threshold
Alert = No otherwise
```

## Sample Output Table

| Timestamp | Person Count | Alert Status |
|---|---:|---|
| 00:00:02 | 1 | No |
| 00:00:04 | 2 | No |
| 00:00:06 | 5 | Yes |

## Screenshots

Add screenshots of the Streamlit UI and annotated output video in the `images/` folder.

## Privacy Note

Do not upload real CCTV footage or private surveillance videos to GitHub. Uploaded videos are processed locally during runtime and saved under the ignored `outputs/` folder. This project is intended as a demonstration and learning project, not as a production surveillance system.

## Limitations

- Accuracy depends on the SSD MobileNet model and camera angle.
- The app samples frames at fixed time intervals, so fast events between sampled frames may be missed.
- The model detects visible people only and may struggle with occlusion, blur, poor lighting, or crowded scenes.
- The application is designed for local demonstration, not real-time multi-camera deployment.

## Future Improvements

- Add optional YOLO-based inference as a documented alternative backend.
- Add charts for crowd count over time.
- Add heatmap-style scene analytics.
- Add multi-video batch processing.
- Add Docker support for reproducible setup.
- Improve output video annotation continuity between sampled frames.

## Author

Harshith Kumar M V

ML / AI Engineer Portfolio Project
