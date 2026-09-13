import json
from pathlib import Path

VIDEO_ANALYSIS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "temp" / "roadvision_latest_analysis.json"
)


def load_video_analysis() -> dict | None:
    """Load the latest video analysis result from disk.

    Always reads fresh from disk, since the file is overwritten on
    every new video analysis and must not be cached at import time.
    """
    if not VIDEO_ANALYSIS_PATH.exists():
        return None

    with VIDEO_ANALYSIS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def build_video_context_block() -> str | None:
    """Build a short text summary of the latest video analysis for the chat prompt."""
    result = load_video_analysis()
    if result is None:
        return None

    # run_video_analysis() wraps the actual analysis dict under "data"
    analysis = result.get("data")
    if not analysis:
        return None

    metadata = analysis.get("video_metadata", "")
    vehicles = analysis.get("vehicle_analysis", "")
    lights = analysis.get("traffic_light_analysis", "")

    lines = [
        f"Video duration: {metadata['duration_seconds']} seconds",
        f"Total unique vehicles detected: {vehicles['total_unique_vehicles']}",
        f"Vehicle breakdown: {vehicles['vehicle_counts']}",
        f"Max active vehicles at once: {vehicles['max_active_vehicles']}",
        f"Traffic lights detected: {lights['total_tracked_traffic_lights']}",
    ]

    return "\n".join(lines)
