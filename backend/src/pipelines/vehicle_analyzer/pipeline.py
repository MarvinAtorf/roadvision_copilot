from pipelines.shared.config import VEHICLE_CLASSES
from pipelines.vehicle_analyzer.config import (
    VEHICLE_EMA_ALPHA,
    VEHICLE_GAP_SECONDS,
    VEHICLE_MATCH_THRESHOLD,
    VEHICLE_NEAR_DISTANCE,
    VEHICLE_SIZE_SIMILARITY,
)
from pipelines.vehicle_analyzer.tracking import (
    deduplicate_frame_detections,
    expire_identities,
    resolve_canonical_id,
)


class VehicleAnalyzer:
    """Tracks and counts vehicles across a video's frames.

    Holds all mutable tracking state internally, so the orchestrator only
    needs to call process_frame() once per frame and build_summary() once
    at the end — it never touches canonical IDs or counters directly.
    """

    def __init__(self):
        self.canonical_tracks: dict[int, dict] = {}
        self.next_id = 1
        self.vehicle_counts = dict.fromkeys(VEHICLE_CLASSES.values(), 0)
        self.total_unique_vehicles = 0
        self.max_active_vehicles = 0

    def process_frame(self, raw_detections, timestamp_seconds, diagonal):
        """Process one frame's detections and return the resolved vehicles.

        raw_detections may contain mixed classes (vehicles and traffic
        lights) — this method filters to vehicle classes itself, so the
        orchestrator does not need to split detections before calling it.
        """
        expire_identities(self.canonical_tracks, timestamp_seconds, VEHICLE_GAP_SECONDS)

        vehicle_only = [d for d in raw_detections if d["class_id"] in VEHICLE_CLASSES]
        detections = deduplicate_frame_detections(vehicle_only)

        resolved = []

        for detection in detections:
            canonical_id, self.next_id, is_new = resolve_canonical_id(
                detection=detection,
                canonical_tracks=self.canonical_tracks,
                next_id=self.next_id,
                current_time=timestamp_seconds,
                diagonal=diagonal,
                match_threshold=VEHICLE_MATCH_THRESHOLD,
                near_distance_threshold=VEHICLE_NEAR_DISTANCE,
                size_similarity_threshold=VEHICLE_SIZE_SIMILARITY,
                ema_alpha=VEHICLE_EMA_ALPHA,
            )

            if is_new:
                self.total_unique_vehicles += 1
                self.vehicle_counts[detection["class_name"]] += 1

            resolved.append({**detection, "canonical_id": canonical_id})

        active_count = len(resolved)
        if active_count > self.max_active_vehicles:
            self.max_active_vehicles = active_count

        return resolved

    def build_summary(self) -> dict:
        """Return the final vehicle_analysis section for the output JSON."""
        return {
            "total_unique_vehicles": self.total_unique_vehicles,
            "vehicle_counts": self.vehicle_counts,
            "max_active_vehicles": self.max_active_vehicles,
        }