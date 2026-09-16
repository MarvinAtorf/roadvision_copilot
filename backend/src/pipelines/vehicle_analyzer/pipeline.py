from pipelines.shared.config import VEHICLE_CLASSES
from pipelines.vehicle_analyzer.config import (
    VEHICLE_CROSS_CLASS_MATCH_THRESHOLD,
    VEHICLE_CROSS_CLASS_MIN_IOU,
    VEHICLE_EMA_ALPHA,
    VEHICLE_GAP_SECONDS,
    VEHICLE_MATCH_THRESHOLD,
    VEHICLE_MIN_AREA_RATIO,
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

    COCO-pretrained YOLO models flicker between "car", "truck" and "bus"
    for high-silhouette vehicles (SUVs, vans, lorries), so a track's
    class is decided by a vote across all the frames it was seen in
    rather than by any single frame. Votes are weighted by confidence
    AND by box size: a vehicle 300 px wide is classified far more
    reliably than the same vehicle at 30 px in the distance, so the
    close-up frames should decide the label instead of being outvoted by
    a long tail of unreliable far-away ones.

    Holds all mutable tracking state internally, so the orchestrator only
    needs to call process_frame() once per frame and build_summary() once
    at the end.
    """

    def __init__(self):
        self.canonical_tracks: dict[int, dict] = {}
        self.next_id = 1
        # canonical_id -> {class_id: accumulated vote weight}
        self.canonical_class_votes: dict[int, dict[int, float]] = {}
        # class_id -> class_name, learned from whatever the model has told us
        self.class_id_to_name: dict[int, str] = {}
        self.max_active_vehicles = 0

    def process_frame(self, raw_detections, timestamp_seconds, diagonal):
        """Process one frame's detections and return the resolved vehicles.

        raw_detections may contain mixed classes (vehicles and traffic
        lights) - this method filters to vehicle classes itself, so the
        orchestrator does not need to split detections before calling it.
        Each resolved detection's class_id/class_name reflect that
        track's current best vote, not necessarily this frame's own raw
        classification.
        """
        expire_identities(self.canonical_tracks, timestamp_seconds, VEHICLE_GAP_SECONDS)

        minimum_area = VEHICLE_MIN_AREA_RATIO * diagonal * diagonal

        vehicle_only = [
            d
            for d in raw_detections
            if d["class_id"] in VEHICLE_CLASSES and d["area"] >= minimum_area
        ]
        detections = deduplicate_frame_detections(vehicle_only)

        resolved = []

        for detection in detections:
            self.class_id_to_name[detection["class_id"]] = detection["class_name"]

            canonical_id, self.next_id, _is_new = resolve_canonical_id(
                detection=detection,
                canonical_tracks=self.canonical_tracks,
                next_id=self.next_id,
                current_time=timestamp_seconds,
                diagonal=diagonal,
                match_threshold=VEHICLE_MATCH_THRESHOLD,
                near_distance_threshold=VEHICLE_NEAR_DISTANCE,
                size_similarity_threshold=VEHICLE_SIZE_SIMILARITY,
                ema_alpha=VEHICLE_EMA_ALPHA,
                cross_class_match_threshold=VEHICLE_CROSS_CLASS_MATCH_THRESHOLD,
                cross_class_min_iou=VEHICLE_CROSS_CLASS_MIN_IOU,
            )

            # sqrt(area) is roughly the box's linear size, so a close
            # vehicle outweighs a distant one in proportion to how much
            # more of it the model actually got to look at.
            weight = detection["confidence"] * (detection["area"] ** 0.5)

            votes = self.canonical_class_votes.setdefault(canonical_id, {})
            votes[detection["class_id"]] = (
                votes.get(detection["class_id"], 0.0) + weight
            )

            best_class_id = max(votes, key=votes.get)
            best_class_name = self.class_id_to_name.get(
                best_class_id, detection["class_name"]
            )

            resolved.append(
                {
                    **detection,
                    "canonical_id": canonical_id,
                    "class_id": best_class_id,
                    "class_name": best_class_name,
                }
            )

        active_count = len(resolved)
        self.max_active_vehicles = max(self.max_active_vehicles, active_count)

        return resolved

    def build_summary(self) -> dict:
        """Return the final vehicle_analysis section for the output JSON.

        Every track's final class is its winning weighted vote across the
        whole video, not whatever it was first seen as.
        """
        vehicle_counts = dict.fromkeys(VEHICLE_CLASSES.values(), 0)

        for votes in self.canonical_class_votes.values():
            best_class_id = max(votes, key=votes.get)
            best_class_name = self.class_id_to_name.get(best_class_id)
            if best_class_name in vehicle_counts:
                vehicle_counts[best_class_name] += 1

        return {
            "total_unique_vehicles": len(self.canonical_class_votes),
            "vehicle_counts": vehicle_counts,
            "max_active_vehicles": self.max_active_vehicles,
        }
