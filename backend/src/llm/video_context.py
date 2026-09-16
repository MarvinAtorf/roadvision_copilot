import json
import re
from pathlib import Path

VIDEO_ANALYSIS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "temp" / "roadvision_latest_analysis.json"
)

# Matches "1:20", "01:20" — minutes:seconds notation.
MINUTES_SECONDS_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")

# Matches "80s", "80 seconds", "80 Sekunden" — plain seconds notation.
PLAIN_SECONDS_PATTERN = re.compile(r"(\d{1,4})\s*(?:s\b|sek(?:unden)?|seconds?)", re.IGNORECASE)


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


def parse_target_seconds(question: str) -> float | None:
    """
    Extract a timestamp mentioned in a chat question, in seconds.

    Supports "1:20" / "01:20" (minutes:seconds) and plain-seconds forms
    like "80s" / "80 seconds" / "80 Sekunden". Returns None if the
    question contains no recognizable timestamp.
    """
    minutes_match = MINUTES_SECONDS_PATTERN.search(question)
    if minutes_match:
        minutes, seconds = minutes_match.groups()
        return int(minutes) * 60 + int(seconds)

    seconds_match = PLAIN_SECONDS_PATTERN.search(question)
    if seconds_match:
        return float(seconds_match.group(1))

    return None


def build_timeline_context_block(question: str) -> str | None:
    """
    Build a compact context block for a timestamp-specific question, e.g.
    "what happens at minute 1:20". Returns None if the question contains
    no timestamp, or if no analysis / timeline data is available yet.

    Looks up the nearest 20s-bucket from timeline_summary (see
    orchestrator.bucket_timeline) rather than the raw per-frame timeline,
    to keep the prompt small and give a stable, deduplicated answer.
    """
    target_seconds = parse_target_seconds(question)
    if target_seconds is None:
        return None

    result = load_video_analysis()
    if result is None:
        return None

    analysis = result.get("data")
    if not analysis:
        return None

    summary = analysis.get("timeline_summary")
    if not summary:
        return None

    duration = analysis["video_metadata"]["duration_seconds"]
    if target_seconds < 0 or target_seconds > duration:
        return (
            f"[video_timeline_out_of_range] The video is only {duration:.0f}s long; "
            f"{target_seconds:.0f}s is outside it."
        )

    nearest_bucket = min(
        summary, key=lambda bucket: abs(bucket["bucket_center_seconds"] - target_seconds)
    )

    window_start, window_end = nearest_bucket["window_seconds"]

    lines = [
        f"[video_timeline_{int(nearest_bucket['bucket_center_seconds'])}s]",
        f"Window {window_start:.0f}-{window_end:.0f}s:",
        f"Vehicles: {nearest_bucket['vehicle_counts']}",
        f"Signs: {nearest_bucket['sign_counts']}",
        f"Traffic light visible: {nearest_bucket['traffic_light_visible']}",
    ]

    return "\n".join(lines)
