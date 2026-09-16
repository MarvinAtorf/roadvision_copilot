from collections import defaultdict

from pipelines.sign_analyzer.config import (
    SIGN_EMA_ALPHA,
    SIGN_GAP_SECONDS,
    SIGN_MATCH_THRESHOLD,
    SIGN_NEAR_DISTANCE,
    SIGN_SIZE_SIMILARITY,
)
from pipelines.sign_analyzer.tracking import (
    deduplicate_frame_detections,
    expire_identities,
    resolve_canonical_id,
)


class SignAnalyzer:
    """Tracks and counts traffic signs across a video's frames.

    A single physical sign gets classified independently on every frame
    it's visible in, and those per-frame classifications don't always
    agree (a distant/blurry frame can misread "40" as "60"). Instead of
    locking a track's label to whatever its first frame said, every
    frame casts a confidence-weighted vote for its track's class, and
    the current leading vote is used as that track's label - so a track
    corrects itself as clearer, closer frames come in, and the final
    summary always reflects the strongest evidence gathered, not the
    first (often worst) glimpse.
    """

    def __init__(self):
        self.canonical_tracks: dict[int, dict] = {}
        self.next_id = 1
        # canonical_id -> {class_id: cumulative confidence}
        self.canonical_class_votes: dict[int, dict[int, float]] = {}
        # class_id -> class_name, learned from whatever the model has told us
        self.class_id_to_name: dict[int, str] = {}
        self.max_active_signs = 0

    def process_frame(self, raw_detections, timestamp_seconds, diagonal):
        """Process one frame's sign detections and return the resolved signs.

        raw_detections is expected to already be sign-only - it comes
        from the separate sign-detection model's own inference pass, so
        no class filtering happens here. Each resolved detection's
        class_id/class_name reflect that track's current best vote, not
        necessarily this frame's own raw classification.
        """
        expire_identities(self.canonical_tracks, timestamp_seconds, SIGN_GAP_SECONDS)

        detections = deduplicate_frame_detections(raw_detections)

        resolved = []

        for detection in detections:
            self.class_id_to_name[detection["class_id"]] = detection["class_name"]

            canonical_id, self.next_id, _is_new = resolve_canonical_id(
                detection=detection,
                canonical_tracks=self.canonical_tracks,
                next_id=self.next_id,
                current_time=timestamp_seconds,
                diagonal=diagonal,
                match_threshold=SIGN_MATCH_THRESHOLD,
                near_distance_threshold=SIGN_NEAR_DISTANCE,
                size_similarity_threshold=SIGN_SIZE_SIMILARITY,
                ema_alpha=SIGN_EMA_ALPHA,
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
        if active_count > self.max_active_signs:
            self.max_active_signs = active_count

        return resolved

    def build_summary(self) -> dict:
        """Return the final traffic_sign_analysis section for the output JSON.

        Every track's final class is its highest-confidence-vote class
        across the whole video, not whatever it was first seen as.
        """
        sign_counts: dict[str, int] = defaultdict(int)

        for votes in self.canonical_class_votes.values():
            best_class_id = max(votes, key=votes.get)
            best_class_name = self.class_id_to_name.get(best_class_id, f"class_{best_class_id}")
            sign_counts[best_class_name] += 1

        return {
            "total_unique_signs": len(self.canonical_class_votes),
            "sign_counts": dict(sign_counts),
            "max_active_signs": self.max_active_signs,
        }