import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_VIDEO = (
    PROJECT_ROOT
    / "backend"
    / "inputs"
    / "videos"
    / "raw_videos"
    / "63621-506830674.mp4"
)

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "backend"
    / "inputs"
    / "videos"
    / "processed_videos"
    / "output_counted.mp4"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "backend"
    / "inputs"
    / "videos"
    / "processed_videos"
    / "video_timeline_analysis.json"
)

MODEL_PATH = PROJECT_ROOT / "backend" / "model_weights" / "yolo11m.pt"


# ============================================================
# YOLO CLASSES
# ============================================================

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}

TRAFFIC_LIGHT_CLASS = 9

TRACKED_CLASSES = [*list(VEHICLE_CLASSES.keys()), TRAFFIC_LIGHT_CLASS]

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# TRACKING PARAMETERS
# ============================================================

# Vehicle identities expire after this many seconds
VEHICLE_GAP_SECONDS = 6.0

# Traffic lights remain available for matching
# for the complete video
TRAFFIC_LIGHT_GAP_SECONDS = None

# Identity matching thresholds
VEHICLE_MATCH_THRESHOLD = 0.40
TRAFFIC_LIGHT_MATCH_THRESHOLD = 0.30

# Spatial matching thresholds
VEHICLE_NEAR_DISTANCE = 0.03
TRAFFIC_LIGHT_NEAR_DISTANCE = 0.05

# Size similarity thresholds
VEHICLE_SIZE_SIMILARITY = 0.40
TRAFFIC_LIGHT_SIZE_SIMILARITY = 0.30

# Strong spatial match
NEAR_SCORE = 0.92

# EMA smoothing
VEHICLE_EMA_ALPHA = 0.60
TRAFFIC_LIGHT_EMA_ALPHA = 0.20

# Remove duplicate detections
DEDUP_IOU_THRESHOLD = 0.60


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def box_center(box):
    """
    Return normalized center coordinates of a bounding box.
    """
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return cx, cy


def box_area(box):
    """
    Calculate bounding-box area.
    """
    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union between two boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def calculate_center_distance(box_a, box_b):
    """
    Calculate normalized Euclidean distance between box centers.
    """

    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def calculate_identity_score(
    current_box,
    previous_box,
    current_area,
    previous_area,
):
    """
    Calculate identity similarity using:
    - IoU
    - center distance
    - bounding-box size similarity
    """

    iou = calculate_iou(
        current_box,
        previous_box,
    )

    center_distance = calculate_center_distance(
        current_box,
        previous_box,
    )

    if previous_area <= 0:
        size_similarity = 0.0
    else:
        area_ratio = current_area / previous_area

        size_similarity = min(
            area_ratio,
            1.0 / area_ratio
            if area_ratio > 0
            else 0.0,
        )

    # Weighted identity score
    score = (
        0.50 * iou
        + 0.30 * (1.0 - min(center_distance, 1.0))
        + 0.20 * size_similarity
    )

    return score


def match_existing_identity(
    current_box,
    candidates,
    image_width,
    image_height,
    match_threshold,
    near_distance_threshold,
    size_similarity_threshold,
):
    """
    Find the best existing canonical identity.

    candidates:
        {
            canonical_id: {
                "box": [...],
                "area": ...,
                ...
            }
        }
    """

    if not candidates:
        return None

    current_area = box_area(current_box)

    best_id = None
    best_score = -1.0

    for canonical_id, identity in candidates.items():

        previous_box = identity["box"]
        previous_area = identity["area"]

        center_distance = calculate_center_distance(
            current_box,
            previous_box,
        )

        area_ratio = (
            current_area / previous_area
            if previous_area > 0
            else 0.0
        )

        size_similarity = min(
            area_ratio,
            1.0 / area_ratio
            if area_ratio > 0
            else 0.0,
        )

        score = calculate_identity_score(
            current_box,
            previous_box,
            current_area,
            previous_area,
        )

        # Strong spatial match
        if (
            center_distance <= near_distance_threshold
            and size_similarity >= size_similarity_threshold
        ):
            return canonical_id

        if (
            score >= match_threshold
            and score > best_score
        ):
            best_score = score
            best_id = canonical_id

    return best_id


def deduplicate_frame_detections(
    detections,
    iou_threshold=DEDUP_IOU_THRESHOLD,
):
    """
    Remove duplicate detections from the same frame.

    Each detection contains:
        {
            "class_id": ...,
            "class_name": ...,
            "confidence": ...,
            "box": [...]
        }
    """

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    kept = []

    for detection in detections:

        duplicate = False

        for existing in kept:

            # Only compare detections of the same class
            if (
                detection["class_id"]
                != existing["class_id"]
            ):
                continue

            iou = calculate_iou(
                detection["box"],
                existing["box"],
            )

            if iou >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


def blend_box(
    previous_box,
    current_box,
    alpha,
):
    """
    Exponential moving average for bounding boxes.
    """

    if previous_box is None:
        return current_box

    return [
        alpha * current_box[i]
        + (1.0 - alpha) * previous_box[i]
        for i in range(4)
    ]


def resolve_canonical_id(
    detection,
    canonical_tracks,
    next_id,
    current_time,
    image_width,
    image_height,
    match_threshold,
    near_distance_threshold,
    size_similarity_threshold,
    ema_alpha,
    max_gap_seconds,
):
    """
    Resolve YOLO/ByteTrack identity into a stable canonical ID.
    """

    current_box = detection["box"]
    current_class_id = detection["class_id"]

    # --------------------------------------------------------
    # Remove expired identities
    # --------------------------------------------------------

    if max_gap_seconds is not None:

        expired_ids = []

        for canonical_id, identity in canonical_tracks.items():

            if (
                current_time
                - identity["last_seen"]
                > max_gap_seconds
            ):
                expired_ids.append(canonical_id)

        for canonical_id in expired_ids:
            del canonical_tracks[canonical_id]

    # --------------------------------------------------------
    # Restrict candidates to same class
    # --------------------------------------------------------

    candidates = {
        canonical_id: identity
        for canonical_id, identity in canonical_tracks.items()
        if identity["class_id"] == current_class_id
    }

    # --------------------------------------------------------
    # Find existing identity
    # --------------------------------------------------------

    matched_id = match_existing_identity(
        current_box=current_box,
        candidates=candidates,
        image_width=image_width,
        image_height=image_height,
        match_threshold=match_threshold,
        near_distance_threshold=near_distance_threshold,
        size_similarity_threshold=size_similarity_threshold,
    )

    # --------------------------------------------------------
    # Existing identity
    # --------------------------------------------------------

    if matched_id is not None:

        identity = canonical_tracks[matched_id]

        identity["box"] = blend_box(
            identity["box"],
            current_box,
            ema_alpha,
        )

        identity["area"] = box_area(
            identity["box"]
        )

        identity["last_seen"] = current_time

        identity["confidence"] = detection[
            "confidence"
        ]

        return matched_id, next_id

    # --------------------------------------------------------
    # Create new identity
    # --------------------------------------------------------

    canonical_id = next_id

    canonical_tracks[canonical_id] = {
        "class_id": current_class_id,
        "class_name": detection["class_name"],
        "box": current_box,
        "area": box_area(current_box),
        "last_seen": current_time,
        "first_seen": current_time,
        "confidence": detection["confidence"],
    }

    next_id += 1

    return canonical_id, next_id


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROADVISION COPILOT - VIDEO ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{INPUT_VIDEO}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(f"\nLoading model:\n{MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{INPUT_VIDEO}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames_in_video = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        raise RuntimeError(
            "Invalid FPS detected."
        )

    video_duration = (
        total_frames_in_video / fps
    )

    print("\nVideo information:")
    print(f"  Resolution: {width} x {height}")
    print(f"  FPS: {fps:.2f}")
    print(
        f"  Frames: {total_frames_in_video}"
    )
    print(
        f"  Duration: {video_duration:.2f} sec"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n"
            f"{OUTPUT_VIDEO}"
        )

    # ========================================================
    # TRACKING STATE
    # ========================================================

    canonical_vehicle_tracks = {}

    canonical_traffic_light_tracks = {}

    next_vehicle_id = 1
    next_traffic_light_id = 1

    vehicle_counts = dict.fromkeys(VEHICLE_CLASSES.values(), 0)

    processed_frames = 0

    max_active_vehicles = 0
    max_visible_traffic_lights = 0

    timeline = []

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        processed_frames += 1

        current_frame_number = processed_frames

        timestamp_seconds = (
            current_frame_number - 1
        ) / fps

        # ----------------------------------------------------
        # YOLO TRACK
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            classes=TRACKED_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        raw_detections = []

        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        if results:

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes

                xyxy = boxes.xyxy.cpu().numpy()

                class_ids = (
                    boxes.cls.cpu().numpy().astype(int)
                )

                confidences = (
                    boxes.conf.cpu().numpy()
                )

                for i in range(len(xyxy)):

                    class_id = class_ids[i]

                    confidence = float(
                        confidences[i]
                    )

                    x1, y1, x2, y2 = xyxy[i]

                    box = [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ]

                    if class_id in VEHICLE_CLASSES:

                        class_name = VEHICLE_CLASSES[
                            class_id
                        ]

                    elif (
                        class_id
                        == TRAFFIC_LIGHT_CLASS
                    ):

                        class_name = "traffic_light"

                    else:
                        continue

                    raw_detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "box": box,
                        }
                    )

        # ----------------------------------------------------
        # Deduplicate same-frame detections
        # ----------------------------------------------------

        detections = (
            deduplicate_frame_detections(
                raw_detections
            )
        )

        # ====================================================
        # CURRENT FRAME DATA
        # ====================================================

        current_vehicle_detections = []
        current_traffic_light_detections = []

        # ====================================================
        # RESOLVE IDENTITIES
        # ====================================================

        for detection in detections:

            class_id = detection[
                "class_id"
            ]

            # ------------------------------------------------
            # VEHICLES
            # ------------------------------------------------

            if class_id in VEHICLE_CLASSES:

                canonical_id, next_vehicle_id = (
                    resolve_canonical_id(
                        detection=detection,
                        canonical_tracks=canonical_vehicle_tracks,
                        next_id=next_vehicle_id,
                        current_time=timestamp_seconds,
                        image_width=width,
                        image_height=height,
                        match_threshold=VEHICLE_MATCH_THRESHOLD,
                        near_distance_threshold=VEHICLE_NEAR_DISTANCE,
                        size_similarity_threshold=VEHICLE_SIZE_SIMILARITY,
                        ema_alpha=VEHICLE_EMA_ALPHA,
                        max_gap_seconds=VEHICLE_GAP_SECONDS,
                    )
                )

                detection["canonical_id"] = (
                    canonical_id
                )

                current_vehicle_detections.append(
                    detection
                )

            # ------------------------------------------------
            # TRAFFIC LIGHTS
            # ------------------------------------------------

            elif class_id == TRAFFIC_LIGHT_CLASS:

                canonical_id, next_traffic_light_id = (
                    resolve_canonical_id(
                        detection=detection,
                        canonical_tracks=canonical_traffic_light_tracks,
                        next_id=next_traffic_light_id,
                        current_time=timestamp_seconds,
                        image_width=width,
                        image_height=height,
                        match_threshold=TRAFFIC_LIGHT_MATCH_THRESHOLD,
                        near_distance_threshold=TRAFFIC_LIGHT_NEAR_DISTANCE,
                        size_similarity_threshold=TRAFFIC_LIGHT_SIZE_SIMILARITY,
                        ema_alpha=TRAFFIC_LIGHT_EMA_ALPHA,
                        max_gap_seconds=TRAFFIC_LIGHT_GAP_SECONDS,
                    )
                )

                detection["canonical_id"] = (
                    canonical_id
                )

                current_traffic_light_detections.append(
                    detection
                )

        # ====================================================
        # UPDATE VEHICLE COUNTS
        # ====================================================

        # Number of active vehicles in current frame
        active_vehicle_count = len(
            current_vehicle_detections
        )

        if active_vehicle_count > max_active_vehicles:
            max_active_vehicles = (
                active_vehicle_count
            )

        # Count unique vehicles by class
        #
        # Because canonical IDs are persistent, we use the
        # identities stored in canonical_vehicle_tracks.
        #
        # This avoids counting the same vehicle multiple times.

        {
            detection["canonical_id"]
            for detection in current_vehicle_detections
        }

        # ----------------------------------------------------
        # Build vehicle class counts from canonical tracks
        # ----------------------------------------------------

        unique_vehicle_class_counts = dict.fromkeys(VEHICLE_CLASSES.values(), 0)

        for identity in canonical_vehicle_tracks.values():

            class_name = identity["class_name"]

            if class_name in unique_vehicle_class_counts:

                unique_vehicle_class_counts[
                    class_name
                ] += 1

        vehicle_counts = (
            unique_vehicle_class_counts
        )

        # ====================================================
        # TRAFFIC LIGHT COUNT
        # ====================================================

        visible_traffic_light_count = len(
            current_traffic_light_detections
        )

        if (
            visible_traffic_light_count
            > max_visible_traffic_lights
        ):
            max_visible_traffic_lights = (
                visible_traffic_light_count
            )

        # ====================================================
        # JSON DETECTION OBJECTS
        # ====================================================

        vehicle_json_detections = []

        for detection in current_vehicle_detections:

            vehicle_json_detections.append(
                {
                    "track_id": int(
                        detection["canonical_id"]
                    ),
                    "class": detection[
                        "class_name"
                    ],
                    "confidence": round(
                        detection["confidence"],
                        4,
                    ),
                    "bbox": [
                        round(value, 2)
                        for value in detection["box"]
                    ],
                }
            )

        traffic_light_json_detections = []

        for detection in current_traffic_light_detections:

            traffic_light_json_detections.append(
                {
                    "track_id": int(
                        detection["canonical_id"]
                    ),
                    "confidence": round(
                        detection["confidence"],
                        4,
                    ),
                    "bbox": [
                        round(value, 2)
                        for value in detection["box"]
                    ],
                }
            )

        # ====================================================
        # TIMELINE
        # ====================================================

        timeline_entry = {
            "timestamp_seconds": round(
                timestamp_seconds,
                3,
            ),

            "frame_number": current_frame_number,

            "traffic_density": {
                "active_vehicles": active_vehicle_count
            },

            "vehicles": {
                "active_count": active_vehicle_count,
                "detections": vehicle_json_detections,
            },

            "traffic_lights": {
                "visible_count": visible_traffic_light_count,
                "detections": (
                    traffic_light_json_detections
                ),
            },

            "traffic_signs": {
                "detections": []
            },
        }

        timeline.append(
            timeline_entry
        )

        # ====================================================
        # DRAW ANNOTATIONS
        # ====================================================

        annotated_frame = frame.copy()

        # ----------------------------------------------------
        # Vehicle boxes
        # ----------------------------------------------------

        for detection in current_vehicle_detections:

            x1, y1, x2, y2 = map(
                int,
                detection["box"],
            )

            track_id = detection[
                "canonical_id"
            ]

            class_name = detection[
                "class_name"
            ]

            confidence = detection[
                "confidence"
            ]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            label = (
                f"{class_name} "
                f"ID:{track_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Traffic light boxes
        # ----------------------------------------------------

        for detection in current_traffic_light_detections:

            x1, y1, x2, y2 = map(
                int,
                detection["box"],
            )

            track_id = detection[
                "canonical_id"
            ]

            confidence = detection[
                "confidence"
            ]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3,
            )

            label = (
                f"traffic_light "
                f"ID:{track_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # ====================================================
        # ====================================================
        # DASHBOARD
        # ====================================================

        dashboard_height = 230

        overlay = annotated_frame[
            :dashboard_height,
            :,
        ].copy()

        cv2.rectangle(
            annotated_frame,
            (0, 0),
            (width, dashboard_height),
            (40, 40, 40),
            -1,
        )

        alpha = 0.80

        annotated_frame[
            :dashboard_height,
            :
        ] = cv2.addWeighted(
            overlay,
            alpha,
            annotated_frame[
                :dashboard_height,
                :
            ],
            1 - alpha,
            0,
        )

        dashboard_lines = [
            (
                f"Frame: "
                f"{current_frame_number}/"
                f"{total_frames_in_video}"
            ),
            (
                f"Time: "
                f"{timestamp_seconds:.2f}s"
            ),
            (
                f"Active vehicles: "
                f"{active_vehicle_count}"
            ),
            (
                f"Unique vehicles: "
                f"{len(canonical_vehicle_tracks)}"
            ),
            (
                f"Cars: "
                f"{vehicle_counts['car']}"
            ),
            (
                f"Trucks: "
                f"{vehicle_counts['truck']}"
            ),
            (
                f"Buses: "
                f"{vehicle_counts['bus']}"
            ),
            (
                f"Motorbikes: "
                f"{vehicle_counts['motorbike']}"
            ),
            (
                f"Bicycles: "
                f"{vehicle_counts['bicycle']}"
            ),
            (
                f"Traffic lights: "
                f"{visible_traffic_light_count}"
            ),
        ]

        y_position = 30

        for line in dashboard_lines:

            cv2.putText(
                annotated_frame,
                line,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            y_position += 22

        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(
            annotated_frame
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            processed_frames % 100 == 0
            or processed_frames
            == total_frames_in_video
        ):

            progress = (
                processed_frames
                / total_frames_in_video
                * 100
                if total_frames_in_video > 0
                else 0
            )

            print(
                f"\rProcessing: "
                f"{processed_frames}/"
                f"{total_frames_in_video} "
                f"({progress:.1f}%)",
                end="",
            )

    print()

    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()
    writer.release()

    # ========================================================
    # FINAL JSON
    # ========================================================

    final_duration_seconds = (
        processed_frames / fps
        if fps > 0
        else 0
    )

    # --------------------------------------------------------
    # Final vehicle counts
    # --------------------------------------------------------

    final_vehicle_counts = dict.fromkeys(VEHICLE_CLASSES.values(), 0)

    for identity in canonical_vehicle_tracks.values():

        class_name = identity[
            "class_name"
        ]

        if class_name in final_vehicle_counts:

            final_vehicle_counts[
                class_name
            ] += 1

    # --------------------------------------------------------
    # JSON structure
    # --------------------------------------------------------

    analysis = {

        "video_metadata": {

            "video_path": str(
                INPUT_VIDEO
            ),

            "fps": round(
                fps,
                3,
            ),

            "width": width,

            "height": height,

            "total_frames_in_video": (
                total_frames_in_video
            ),

            "frames_processed": (
                processed_frames
            ),

            "duration_seconds": round(
                final_duration_seconds,
                3,
            ),
        },

        "vehicle_analysis": {

            "total_unique_vehicles": len(
                canonical_vehicle_tracks
            ),

            "vehicle_counts": (
                final_vehicle_counts
            ),

            "max_active_vehicles": (
                max_active_vehicles
            ),
        },

        "traffic_light_analysis": {

            "total_tracked_traffic_lights": len(
                canonical_traffic_light_tracks
            ),

            "max_visible_simultaneously": (
                max_visible_traffic_lights
            ),
        },

        "traffic_sign_analysis": {

            "status": (
                "not_implemented_in_mvp"
            ),

            "detections": [],
        },

        "timeline": timeline,
    }

    # ========================================================
    # WRITE JSON
    # ========================================================

    with Path.open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            analysis,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nProcessed frames: "
        f"{processed_frames}"
    )

    print(
        f"Unique vehicles: "
        f"{len(canonical_vehicle_tracks)}"
    )

    print(
        f"Max active vehicles: "
        f"{max_active_vehicles}"
    )

    print(
        f"Tracked traffic lights: "
        f"{len(canonical_traffic_light_tracks)}"
    )

    print(
        f"Max visible traffic lights: "
        f"{max_visible_traffic_lights}"
    )

    print(
        f"\nOutput video:\n"
        f"{OUTPUT_VIDEO}"
    )

    print(
        f"\nOutput JSON:\n"
        f"{OUTPUT_JSON}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
