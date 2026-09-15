import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX

DASHBOARD_HEIGHT = 230
DASHBOARD_ALPHA = 0.80

DASHBOARD_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)

# One color per detected vehicle class.
# OpenCV uses BGR format.
VEHICLE_COLORS = {
    "car": (0, 255, 0),
    "truck": (255, 165, 0),
    "bus": (255, 0, 255),
    "motorbike": (0, 165, 255),
    "bicycle": (255, 255, 0),
}


def draw_vehicle_annotations(
    frame,
    vehicle_detections,
):
    """
    Draw bounding boxes and labels for detected vehicles.

    Each detection is expected to contain:
        {
            "class_name": str,
            "canonical_id": int,
            "confidence": float,
            "box": [x1, y1, x2, y2],
        }
    """

    annotated_frame = frame

    for detection in vehicle_detections:
        x1, y1, x2, y2 = map(
            int,
            detection["box"],
        )

        class_name = detection["class_name"]
        track_id = detection["canonical_id"]
        confidence = detection["confidence"]

        color = VEHICLE_COLORS.get(
            class_name,
            (0, 255, 0),
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        label = (
            f"{class_name} "
            f"ID:{track_id} "
            f"{confidence:.2f}"
        )

        label_y = max(
            30,
            y1 - 10,
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, label_y),
            FONT,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


def draw_vehicle_dashboard(
    frame,
    current_frame_number,
    total_frames,
    timestamp_seconds,
    active_vehicle_count,
    unique_vehicle_count,
    vehicle_counts,
    traffic_light_count,
):
    """
    Draw the vehicle analysis dashboard.

    vehicle_counts should contain:
        car
        truck
        bus
        motorbike
        bicycle
    """

    height, width = frame.shape[:2]

    dashboard_height = min(
        DASHBOARD_HEIGHT,
        height,
    )

    overlay = frame[
        :dashboard_height,
        :,
    ].copy()

    cv2.rectangle(
        frame,
        (0, 0),
        (width, dashboard_height),
        DASHBOARD_COLOR,
        -1,
    )

    frame[:dashboard_height, :] = cv2.addWeighted(
        overlay,
        DASHBOARD_ALPHA,
        frame[:dashboard_height, :],
        1.0 - DASHBOARD_ALPHA,
        0,
    )

    dashboard_lines = [
        (
            f"Frame: "
            f"{current_frame_number}/{total_frames}"
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
            f"{unique_vehicle_count}"
        ),
        (
            f"Traffic lights: "
            f"{traffic_light_count}"
        ),
        (
            f"Cars: "
            f"{vehicle_counts.get('car', 0)}"
        ),
        (
            f"Trucks: "
            f"{vehicle_counts.get('truck', 0)}"
        ),
        (
            f"Buses: "
            f"{vehicle_counts.get('bus', 0)}"
        ),
        (
            f"Motorbikes: "
            f"{vehicle_counts.get('motorbike', 0)}"
        ),
        (
            f"Bicycles: "
            f"{vehicle_counts.get('bicycle', 0)}"
        ),
    ]

    y_position = 30

    for line in dashboard_lines:
        cv2.putText(
            frame,
            line,
            (20, y_position),
            FONT,
            0.75,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        y_position += 22

    return frame


def annotate_vehicle_frame(
    frame,
    vehicle_detections,
    current_frame_number,
    total_frames,
    timestamp_seconds,
    active_vehicle_count,
    unique_vehicle_count,
    vehicle_counts,
    traffic_light_count,
):
    """
    Apply all vehicle-related annotations to one frame.
    """

    frame = draw_vehicle_annotations(
        frame,
        vehicle_detections,
    )

    frame = draw_vehicle_dashboard(
        frame,
        current_frame_number,
        total_frames,
        timestamp_seconds,
        active_vehicle_count,
        unique_vehicle_count,
        vehicle_counts,
        traffic_light_count,
    )

    return frame