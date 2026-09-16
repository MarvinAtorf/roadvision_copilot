import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

DASHBOARD_HEIGHT = 230
DASHBOARD_ALPHA = 0.80
DASHBOARD_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)

# One color per detected class, so vehicles of different types are
# visually distinguishable on the annotated frame, not just traffic
# lights vs. vehicles.
CLASS_COLORS = {
    "car": (0, 255, 0),  # green
    "truck": (0, 165, 255),  # orange
    "bus": (255, 0, 255),  # magenta
    "motorbike": (255, 255, 0),  # cyan
    "bicycle": (0, 0, 255),  # red
    "traffic_light": (0, 255, 255),  # yellow
}

# Traffic signs can be dozens of distinct classes (GTSDB taxonomy), so
# instead of assigning each one its own color, every detection tagged
# with this category gets one shared, distinct color.
CATEGORY_COLORS = {
    "traffic_sign": (0, 140, 255),  # orange-red
}

DEFAULT_COLOR = (200, 200, 200)  # gray fallback for unmapped classes


def draw_detections(frame, detections):
    """Draw bounding boxes and labels, colored per detected class.

    Detections carrying a "category" key (e.g. "traffic_sign") are
    colored by category instead of by their specific class_name, since
    some categories have too many classes to color individually.
    """
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["box"])

        class_name = detection["class_name"]
        category = detection.get("category")
        color = CATEGORY_COLORS.get(category) or CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{class_name} ID:{detection['canonical_id']} {detection['confidence']:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            FONT,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_dashboard(frame, width, lines):
    """Draw the semi-transparent statistics dashboard."""
    region = frame[:DASHBOARD_HEIGHT, :]

    overlay = region.copy()
    cv2.rectangle(overlay, (0, 0), (width, DASHBOARD_HEIGHT), DASHBOARD_COLOR, -1)
    cv2.addWeighted(region, DASHBOARD_ALPHA, overlay, 1.0 - DASHBOARD_ALPHA, 0, dst=region)

    y_position = 30

    for line in lines:
        cv2.putText(frame, line, (20, y_position), FONT, 0.75, TEXT_COLOR, 2, cv2.LINE_AA)
        y_position += 22
