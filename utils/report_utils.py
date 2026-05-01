import json


def convert_results_to_csv(df):
    """Export the timestamp-wise DataFrame as downloadable CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def convert_results_to_json(results):
    """Export detection annotations as a downloadable JSON report."""
    return json.dumps(results, indent=2).encode("utf-8")


def generate_summary_report(summary):
    """Build a readable text summary for the Streamlit report panel."""
    lines = [
        "CCTV Person Detection Summary",
        "",
        f"Total frames processed: {summary['total_frames_processed']}",
        f"Total timestamps analyzed: {summary['total_timestamps_analyzed']}",
        f"Maximum person count: {summary['max_person_count']}",
        f"Minimum person count: {summary['min_person_count']}",
        f"Average person count: {summary['average_person_count']}",
        f"Peak crowd timestamp: {summary['peak_crowd_timestamp']}",
        f"Alert events: {summary['alert_count']}",
    ]
    return "\n".join(lines)
