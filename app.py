import os

# Keep this before TensorFlow is imported by detection.py.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pathlib import Path

import streamlit as st

from detection import process_video
from utils.report_utils import (
    convert_results_to_csv,
    convert_results_to_json,
    generate_summary_report,
)
from utils.video_utils import save_uploaded_video


st.set_page_config(
    page_title="AI-Powered CCTV Person Detection",
    layout="wide",
)


st.title("AI-Powered CCTV Person Detection and Crowd Monitoring System")

st.write(
    "Upload CCTV footage, run TensorFlow SSD MobileNet person detection, "
    "monitor crowd levels by timestamp, and export detection reports."
)


with st.sidebar:
    st.header("Detection Settings")

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05,
        help="Only person detections at or above this score are counted.",
    )

    frame_interval_seconds = st.selectbox(
        "Frame interval",
        options=[1, 2, 5],
        index=2,
        format_func=lambda value: f"Every {value} second{'s' if value > 1 else ''}",
        help=(
            "Higher interval is faster. For example, every 5 seconds processes "
            "fewer frames than every 1 or 2 seconds."
        ),
    )

    alert_threshold = st.slider(
        "Crowd alert threshold",
        min_value=1,
        max_value=25,
        value=3,
        step=1,
        help="An alert is generated when person count reaches this number.",
    )

    st.divider()

    generate_processed_video = st.checkbox(
        "Generate processed annotated video",
        value=False,
        help=(
            "Keep this OFF for faster results. Turn ON only when you need an "
            "annotated output video with bounding boxes."
        ),
    )

    st.info(
        "Fast mode is enabled when annotated video generation is OFF. "
        "This generates summary metrics, table, CSV, and JSON much faster."
    )


uploaded_file = st.file_uploader(
    "Upload CCTV video",
    type=["mp4", "avi", "mov", "mkv"],
)


if uploaded_file is not None:
    input_video_path = save_uploaded_video(uploaded_file)

    st.subheader("Input Video Preview")
    st.video(str(input_video_path))

    st.caption(
        "Tip: For quick testing, keep frame interval as 5 seconds and keep "
        "'Generate processed annotated video' turned OFF."
    )

    run_detection = st.button("Run Detection", type="primary")

    if run_detection:
        try:
            st.info(
                "Detection started. Please wait. "
                "If annotated video generation is OFF, results should generate faster."
            )

            with st.spinner("Running TensorFlow person detection..."):
                df, results, summary, output_video_path = process_video(
                    str(input_video_path),
                    confidence_threshold=confidence_threshold,
                    frame_interval_seconds=frame_interval_seconds,
                    alert_threshold=alert_threshold,
                    save_output_video=generate_processed_video,
                )

            st.success("Detection completed successfully.")

            st.subheader("Detection Summary")

            metric_columns = st.columns(4)

            metric_columns[0].metric(
                "Total Frames Processed",
                summary["total_frames_processed"],
            )

            metric_columns[1].metric(
                "Timestamps Analyzed",
                summary["total_timestamps_analyzed"],
            )

            metric_columns[2].metric(
                "Maximum Person Count",
                summary["max_person_count"],
            )

            metric_columns[3].metric(
                "Average Person Count",
                summary["average_person_count"],
            )

            metric_columns = st.columns(3)

            metric_columns[0].metric(
                "Peak Crowd Timestamp",
                summary["peak_crowd_timestamp"],
            )

            metric_columns[1].metric(
                "Alert Events",
                summary["alert_count"],
            )

            metric_columns[2].metric(
                "Minimum Person Count",
                summary["min_person_count"],
            )

            if generate_processed_video:
                if output_video_path and Path(output_video_path).exists():
                    st.subheader("Processed Video Preview")
                    st.video(output_video_path)
                else:
                    st.warning(
                        "Processed video was requested, but no output video was generated."
                    )
            else:
                st.info(
                    "Processed video generation was skipped to improve speed. "
                    "Turn ON 'Generate processed annotated video' in the sidebar if needed."
                )

            st.subheader("Timestamp-Wise Detection Table")

            if df.empty:
                st.info("No timestamps were analyzed. Try a shorter frame interval.")
            else:
                st.dataframe(df, width="stretch")

                alert_rows = df[df["Alert Status"] == "Yes"]

                if not alert_rows.empty:
                    st.subheader("Alert Events")
                    st.dataframe(alert_rows, width="stretch")
                else:
                    st.info("No crowd alert events were detected.")

            with st.expander("Summary Report"):
                st.text(generate_summary_report(summary))

            csv_report = convert_results_to_csv(df)
            json_report = convert_results_to_json(results)

            download_columns = st.columns(2)

            download_columns[0].download_button(
                "Download CSV Report",
                data=csv_report,
                file_name="cctv_person_detection_report.csv",
                mime="text/csv",
            )

            download_columns[1].download_button(
                "Download JSON Report",
                data=json_report,
                file_name="cctv_person_detection_report.json",
                mime="application/json",
            )

        except FileNotFoundError as error:
            st.error(str(error))
            st.info(
                "Copy the original SSD MobileNet frozen graph into "
                "`models/ssd_mobilenet_v1_coco_2018_01_28/` or set "
                "`CCTV_MODEL_PATH` before running Streamlit."
            )

        except TypeError as error:
            st.error(
                "The current detection.py may not support the save_output_video option."
            )
            st.exception(error)
            st.info(
                "Open detection.py and confirm that process_video has this parameter: "
                "`save_output_video: bool = True`."
            )

        except Exception as error:
            st.error("Detection failed. Check the video file and model setup.")
            st.exception(error)

else:
    st.info("Upload a CCTV video to begin.")
