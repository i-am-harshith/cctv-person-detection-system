from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"


def ensure_output_dir() -> Path:
    """Create the local output directory used for uploads and reports."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def format_timestamp(seconds: float) -> str:
    """Convert seconds into HH:MM:SS format for timestamp-wise reporting."""
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_uploaded_video(uploaded_file) -> Path:
    """Save a Streamlit uploaded video locally before OpenCV processing."""
    output_dir = ensure_output_dir()
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    safe_stem = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in Path(uploaded_file.name).stem
    )
    output_path = output_dir / f"uploaded_{safe_stem}{suffix}"

    with output_path.open("wb") as file_handle:
        file_handle.write(uploaded_file.getbuffer())

    return output_path
