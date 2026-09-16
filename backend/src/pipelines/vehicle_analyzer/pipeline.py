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
 
    COCO-pretrained YOLO models are known to flicker between "car" and
    "truck" for high-silhouette vehicles (SUVs, vans, MPVs) frame to
    frame. Locking a track's class to whatever its first frame said (or
    worse, treating each class flip as a brand-new track) causes real
    double-counting. Instead, every frame casts a confidence-weighted
    vote for its track's class, and the current leading vote is used as
    that track's label - so a track's final class reflects the
    strongest evidence gathered across its whole lifetime, not a single
    frame's guess.
 
    Holds all mutable tracking state internally, so the orchestrator only
    needs to call process_frame() once per frame and build_summary() once
    at the end.
    """
 
    def __init__(self):
        self.canonical_tracks: dict[int, dict] = {}
        self.next_id = 1
        # canonical_id -> {class_id: cumulative confidence}
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
 
        vehicle_only = [d for d in raw_detections if d["class_id"] in VEHICLE_CLASSES]
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
            )
 
            votes = self.canonical_class_votes.setdefault(canonical_id, {})
            votes[detection["class_id"]] = votes.get(detection["class_id"], 0.0) + detection[
                "confidence"
            ]
 
            best_class_id = max(votes, key=votes.get)
            best_class_name = self.class_id_to_name.get(best_class_id, detection["class_name"])
 
            resolved.append(
                {
                    **detection,
                    "canonical_id": canonical_id,
                    "class_id": best_class_id,
                    "class_name": best_class_name,
                }
            )
 
        active_count = len(resolved)
        if active_count > self.max_active_vehicles:
            self.max_active_vehicles = active_count
 
        return resolved
 
    def build_summary(self) -> dict:
        """Return the final vehicle_analysis section for the output JSON.
 
        Every track's final class is its highest-confidence-vote class
        across the whole video, not whatever it was first seen as - this
        is what actually fixes car/truck double-counting for the same
        physical vehicle.
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
 
