"""
Thread-safe shared state for the video analysis currently in progress.

Written to by orchestrator.process_video() after every processed frame,
read by the /analyze/video/status endpoint while the analysis runs in a
background thread.
"""

import threading

from pydantic import BaseModel, computed_field


class LiveStatus(BaseModel):
    """Live progress and cumulative counts for the running analysis."""

    frame_number: int
    total_frames_in_video: int
    timestamp_seconds: float
    vehicle_counts: dict[str, int]
    total_unique_vehicles: int
    traffic_light_count: int

    @computed_field
    @property
    def progress_percent(self) -> float:
        if self.total_frames_in_video == 0:
            return 0.0
        return round(100 * self.frame_number / self.total_frames_in_video, 1)


_lock = threading.Lock()
_status: dict = {}


def reset_status() -> None:
    """Clear the status before starting a new video analysis."""
    with _lock:
        _status.clear()


def update_status(**fields) -> None:
    """Update one or more fields of the shared status."""
    with _lock:
        _status.update(fields)


def get_status() -> LiveStatus | None:
    """Return the current status, or None if no analysis is in progress."""
    with _lock:
        if not _status:
            return None
        return LiveStatus(**_status)
