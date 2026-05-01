import os

# Keep this before TensorFlow import for old Object Detection protobuf compatibility.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

from utils import people_class_util as class_utils
from utils import visualization_utils as vis_util
from utils.video_utils import ensure_output_dir, format_timestamp


# Required because this project uses old TensorFlow 1.x frozen graph logic.
tf.compat.v1.disable_eager_execution()


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ssd_mobilenet_v1_coco_2018_01_28"
    / "frozen_inference_graph.pb"
)

DEFAULT_CATEGORY_INDEX = {
    1: {
        "id": 1,
        "name": "person",
    }
}


def load_detection_graph(model_path: Optional[str] = None) -> tf.Graph:
    """Load the TensorFlow SSD MobileNet frozen inference graph."""
    selected_model_path = Path(
        model_path or os.getenv("CCTV_MODEL_PATH") or DEFAULT_MODEL_PATH
    )

    if not selected_model_path.exists():
        raise FileNotFoundError(
            "TensorFlow frozen graph not found. Place frozen_inference_graph.pb "
            f"at {DEFAULT_MODEL_PATH} or set CCTV_MODEL_PATH to the model file."
        )

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(str(selected_model_path), "rb") as file_handle:
            serialized_graph = file_handle.read()
            graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(graph_def, name="")

    return detection_graph


def load_category_index(label_map_path: Optional[str] = None) -> Dict[int, Dict[str, str]]:
    """Load category labels. Defaults to person-only class mapping."""
    if not label_map_path:
        return DEFAULT_CATEGORY_INDEX

    from utils import label_map_util

    label_path = Path(label_map_path)

    if not label_path.exists():
        return DEFAULT_CATEGORY_INDEX

    return label_map_util.create_category_index_from_labelmap(str(label_path))


def _get_tensor_dict(detection_graph: tf.Graph) -> Dict[str, tf.Tensor]:
    """Fetch TensorFlow tensors once and reuse them for all sampled frames."""
    return {
        "image_tensor": detection_graph.get_tensor_by_name("image_tensor:0"),
        "boxes": detection_graph.get_tensor_by_name("detection_boxes:0"),
        "scores": detection_graph.get_tensor_by_name("detection_scores:0"),
        "classes": detection_graph.get_tensor_by_name("detection_classes:0"),
        "num_detections": detection_graph.get_tensor_by_name("num_detections:0"),
    }


def run_inference_on_frame(
    session: tf.compat.v1.Session,
    frame: np.ndarray,
    tensor_dict: Dict[str, tf.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run object detection inference on one frame."""
    frame_batch = np.expand_dims(frame, axis=0)

    boxes, scores, classes, num_detections = session.run(
        [
            tensor_dict["boxes"],
            tensor_dict["scores"],
            tensor_dict["classes"],
            tensor_dict["num_detections"],
        ],
        feed_dict={tensor_dict["image_tensor"]: frame_batch},
    )

    return (
        np.squeeze(boxes),
        np.squeeze(scores),
        np.squeeze(classes).astype(np.int32),
        int(np.squeeze(num_detections)),
    )


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    category_index: Dict[int, Dict[str, str]],
    confidence_threshold: float,
    timestamp: str,
    person_count: int,
    alert_status: str,
) -> np.ndarray:
    """Draw bounding boxes and overlay timestamp/person-count info."""
    annotated_frame = frame.copy()

    vis_util.visualize_boxes_and_labels_on_image_array(
        annotated_frame,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=confidence_threshold,
        line_thickness=3,
        draw_only_person=True,
    )

    cv2.putText(
        annotated_frame,
        f"{timestamp} | Persons: {person_count} | Alert: {alert_status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if alert_status == "Yes" else (0, 180, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated_frame


def _build_summary(
    results: List[Dict],
    total_frames_processed: int,
) -> Dict[str, object]:
    """Create crowd-monitoring summary metrics."""
    counts = [result["person_count"] for result in results]
    alert_count = sum(1 for result in results if result["alert_status"] == "Yes")

    if counts:
        max_person_count = max(counts)
        peak_index = counts.index(max_person_count)
        peak_crowd_timestamp = results[peak_index]["timestamp"]
        min_person_count = min(counts)
        average_person_count = round(float(np.mean(counts)), 2)
    else:
        max_person_count = 0
        min_person_count = 0
        average_person_count = 0.0
        peak_crowd_timestamp = "N/A"

    return {
        "total_frames_processed": total_frames_processed,
        "total_timestamps_analyzed": len(results),
        "max_person_count": max_person_count,
        "min_person_count": min_person_count,
        "average_person_count": average_person_count,
        "peak_crowd_timestamp": peak_crowd_timestamp,
        "alert_count": alert_count,
    }


def _create_sample_frame_numbers(
    total_frames: int,
    fps: float,
    frame_interval_seconds: int,
) -> List[int]:
    """Create frame positions to sample directly instead of reading every frame."""
    sample_every_frames = max(1, int(round(fps * frame_interval_seconds)))

    # Start at first completed interval: 5s, 10s, 15s, etc.
    sample_frames = list(range(sample_every_frames, total_frames + 1, sample_every_frames))

    # If video is shorter than the selected interval, process the last frame once.
    if not sample_frames and total_frames > 0:
        sample_frames = [total_frames]

    return sample_frames


def process_video(
    video_path: str,
    confidence_threshold: float,
    frame_interval_seconds: int,
    alert_threshold: int,
    model_path: Optional[str] = None,
    save_output_video: bool = False,
) -> Tuple[pd.DataFrame, List[Dict], Dict[str, object], Optional[str]]:
    """Process CCTV video and return timestamp-wise person-count analytics.

    Optimized version:
    - Does not read every frame sequentially.
    - Jumps directly to sampled timestamps.
    - Skips annotation/video writing when save_output_video=False.
    - Much faster for Streamlit use.
    """
    detection_graph = load_detection_graph(model_path=model_path)
    category_index = load_category_index()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_frame_numbers = _create_sample_frame_numbers(
        total_frames=total_frames,
        fps=fps,
        frame_interval_seconds=frame_interval_seconds,
    )

    output_path = None
    writer = None

    if save_output_video and frame_width > 0 and frame_height > 0:
        output_dir = ensure_output_dir()
        output_path = output_dir / f"annotated_{Path(video_path).stem}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Output video contains only sampled annotated frames, so keep FPS low.
        output_fps = 1.0

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (frame_width, frame_height),
        )

        if not writer.isOpened():
            writer = None
            output_path = None

    results: List[Dict] = []
    table_rows: List[Dict[str, object]] = []
    processed_frames_count = 0

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as session:
            tensor_dict = _get_tensor_dict(detection_graph)

            for frame_number in sample_frame_numbers:
                # OpenCV frame index is 0-based.
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))

                success, frame = capture.read()

                if not success or frame is None:
                    continue

                processed_frames_count += 1

                timestamp_seconds = frame_number / fps
                timestamp = format_timestamp(timestamp_seconds)

                print(f"Processing timestamp: {timestamp}")

                boxes, scores, classes, _ = run_inference_on_frame(
                    session=session,
                    frame=frame,
                    tensor_dict=tensor_dict,
                )

                annotations, person_count = class_utils.get_class(
                    classes,
                    category_index,
                    boxes,
                    scores,
                    min_score_thresh=confidence_threshold,
                )

                alert_status = "Yes" if person_count >= alert_threshold else "No"

                result = {
                    "timestamp": timestamp,
                    "timestamp_seconds": round(float(timestamp_seconds), 2),
                    "person_count": person_count,
                    "alert_status": alert_status,
                    "class_annotations": annotations,
                }

                results.append(result)

                table_rows.append(
                    {
                        "Timestamp": timestamp,
                        "Person Count": person_count,
                        "Alert Status": alert_status,
                    }
                )

                if writer is not None:
                    annotated_frame = draw_detections(
                        frame=frame,
                        boxes=boxes,
                        scores=scores,
                        classes=classes,
                        category_index=category_index,
                        confidence_threshold=confidence_threshold,
                        timestamp=timestamp,
                        person_count=person_count,
                        alert_status=alert_status,
                    )

                    writer.write(annotated_frame)

    capture.release()

    if writer is not None:
        writer.release()

    dataframe = pd.DataFrame(table_rows)

    summary = _build_summary(
        results=results,
        total_frames_processed=processed_frames_count,
    )

    return dataframe, results, summary, str(output_path) if output_path else None
