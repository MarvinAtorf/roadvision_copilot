from pipelines.shared.config import TRAFFIC_LIGHT_CLASS
from pipelines.traffic_light_analyzer.config import (
    TRAFFIC_LIGHT_EMA_ALPHA,
    TRAFFIC_LIGHT_GAP_SECONDS,
    TRAFFIC_LIGHT_MATCH_THRESHOLD,
    TRAFFIC_LIGHT_NEAR_DISTANCE,
    TRAFFIC_LIGHT_SIZE_SIMILARITY,
)
from pipelines.traffic_light_analyzer.tracking import (
    deduplicate_frame_detections,
    expire_identities,
    resolve_canonical_id,
)


class TrafficLightAnalyzer:
    """Tracks traffic lights across a video's frames.

    Same role as VehicleAnalyzer, but simpler: there is only one class
    ("traffic_light"), so no per-class breakdown is needed.
    """

    def __init__(self):
        self.canonical_tracks: dict[int, dict] = {}
        self.next_id = 1
        self.max_visible_simultaneously = 0

    def process_frame(self, raw_detections, timestamp_seconds, diagonal):
        """Process one frame's detections and return the resolved traffic lights.

        raw_detections may contain mixed classes (vehicles and traffic
        lights) — this method filters to traffic lights itself, so the
        orchestrator does not need to split detections before calling it.
        """
        expire_identities(self.canonical_tracks, timestamp_seconds, TRAFFIC_LIGHT_GAP_SECONDS)

        light_only = [d for d in raw_detections if d["class_id"] == TRAFFIC_LIGHT_CLASS]
        detections = deduplicate_frame_detections(light_only)

        resolved = []

        for detection in detections:
            canonical_id, self.next_id, _ = resolve_canonical_id(
                detection=detection,
                canonical_tracks=self.canonical_tracks,
                next_id=self.next_id,
                current_time=timestamp_seconds,
                diagonal=diagonal,
                match_threshold=TRAFFIC_LIGHT_MATCH_THRESHOLD,
                near_distance_threshold=TRAFFIC_LIGHT_NEAR_DISTANCE,
                size_similarity_threshold=TRAFFIC_LIGHT_SIZE_SIMILARITY,
                ema_alpha=TRAFFIC_LIGHT_EMA_ALPHA,
            )

            resolved.append({**detection, "canonical_id": canonical_id})

        visible_count = len(resolved)
        if visible_count > self.max_visible_simultaneously:
            self.max_visible_simultaneously = visible_count

        return resolved

    def build_summary(self) -> dict:
        """Return the final traffic_light_analysis section for the output JSON."""
        return {
            "total_tracked_traffic_lights": self.next_id - 1,
            "max_visible_simultaneously": self.max_visible_simultaneously,
        }