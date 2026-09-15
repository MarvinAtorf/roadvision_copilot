from pipelines.shared.geometry import box_area, box_center, calculate_iou, size_similarity_ratio
from pipelines.vehicle_analyzer.config import (
    DEDUP_IOU_THRESHOLD,
    SCORE_WEIGHT_DISTANCE,
    SCORE_WEIGHT_IOU,
    SCORE_WEIGHT_SIZE,
)


def match_existing_identity(
    current_box,
    current_area,
    candidates,
    diagonal,
    match_threshold,
    near_distance_threshold,
    size_similarity_threshold,
):
    """Find the best existing canonical identity for a detection.

    All three metrics (IoU, center distance, size similarity) are computed
    exactly once per candidate and then combined, instead of being
    recalculated inside a separate scoring helper.
    """
    if not candidates:
        return None

    current_cx, current_cy = box_center(current_box)

    best_id = None
    best_score = -1.0

    for canonical_id, identity in candidates.items():
        previous_box = identity["box"]
        previous_area = identity["area"]

        previous_cx, previous_cy = box_center(previous_box)

        # Normalized by the image diagonal so the thresholds are
        # resolution independent.
        dx = current_cx - previous_cx
        dy = current_cy - previous_cy
        center_distance = ((dx * dx + dy * dy) ** 0.5) / diagonal

        size_similarity = size_similarity_ratio(current_area, previous_area)

        # Strong spatial match short-circuits the search.
        if (
            center_distance <= near_distance_threshold
            and size_similarity >= size_similarity_threshold
        ):
            return canonical_id

        iou = calculate_iou(current_box, previous_box, current_area, previous_area)

        score = (
            SCORE_WEIGHT_IOU * iou
            + SCORE_WEIGHT_DISTANCE * (1.0 - min(center_distance, 1.0))
            + SCORE_WEIGHT_SIZE * size_similarity
        )

        if score >= match_threshold and score > best_score:
            best_score = score
            best_id = canonical_id

    return best_id


def deduplicate_frame_detections(detections, iou_threshold=DEDUP_IOU_THRESHOLD):
    """Remove duplicate detections from the same frame."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    kept = []

    for detection in detections:
        box = detection["box"]
        area = detection["area"]
        class_id = detection["class_id"]

        duplicate = False

        for existing in kept:
            if class_id != existing["class_id"]:
                continue

            if calculate_iou(box, existing["box"], area, existing["area"]) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


def blend_box(previous_box, current_box, alpha):
    """Exponential moving average for bounding boxes."""
    if previous_box is None:
        return current_box

    inverse_alpha = 1.0 - alpha

    return [
        alpha * current_box[0] + inverse_alpha * previous_box[0],
        alpha * current_box[1] + inverse_alpha * previous_box[1],
        alpha * current_box[2] + inverse_alpha * previous_box[2],
        alpha * current_box[3] + inverse_alpha * previous_box[3],
    ]


def expire_identities(canonical_tracks, current_time, max_gap_seconds):
    """Drop identities that have not been seen for too long."""
    if max_gap_seconds is None or not canonical_tracks:
        return

    expired = [
        canonical_id
        for canonical_id, identity in canonical_tracks.items()
        if current_time - identity["last_seen"] > max_gap_seconds
    ]

    for canonical_id in expired:
        del canonical_tracks[canonical_id]


def resolve_canonical_id(
    detection,
    canonical_tracks,
    next_id,
    current_time,
    diagonal,
    match_threshold,
    near_distance_threshold,
    size_similarity_threshold,
    ema_alpha,
):
    """Resolve a raw detection into a stable canonical ID.

    Returns (canonical_id, next_id, is_new_identity).
    """
    current_box = detection["box"]
    current_area = detection["area"]
    current_class_id = detection["class_id"]

    candidates = {
        canonical_id: identity
        for canonical_id, identity in canonical_tracks.items()
        if identity["class_id"] == current_class_id
    }

    matched_id = match_existing_identity(
        current_box=current_box,
        current_area=current_area,
        candidates=candidates,
        diagonal=diagonal,
        match_threshold=match_threshold,
        near_distance_threshold=near_distance_threshold,
        size_similarity_threshold=size_similarity_threshold,
    )

    if matched_id is not None:
        identity = canonical_tracks[matched_id]

        identity["box"] = blend_box(identity["box"], current_box, ema_alpha)
        identity["area"] = box_area(identity["box"])
        identity["last_seen"] = current_time
        identity["confidence"] = detection["confidence"]

        return matched_id, next_id, False

    canonical_tracks[next_id] = {
        "class_id": current_class_id,
        "class_name": detection["class_name"],
        "box": current_box,
        "area": current_area,
        "last_seen": current_time,
        "first_seen": current_time,
        "confidence": detection["confidence"],
    }

    return next_id, next_id + 1, True
