def box_center(box):
    """Return the center coordinates of a bounding box in pixels."""
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def box_area(box):
    """Calculate bounding-box area."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    if width <= 0.0 or height <= 0.0:
        return 0.0

    return width * height


def calculate_iou(box_a, box_b, area_a=None, area_b=None):
    """Calculate Intersection over Union between two boxes.

    Areas can be passed in when they are already known, which avoids
    recomputing them for every candidate comparison.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = min(ax2, bx2) - max(ax1, bx1)
    if intersection_width <= 0.0:
        return 0.0

    intersection_height = min(ay2, by2) - max(ay1, by1)
    if intersection_height <= 0.0:
        return 0.0

    intersection_area = intersection_width * intersection_height

    if area_a is None:
        area_a = box_area(box_a)
    if area_b is None:
        area_b = box_area(box_b)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def size_similarity_ratio(current_area, previous_area):
    """Return a 0..1 similarity between two areas (1 = identical size)."""
    if previous_area <= 0.0 or current_area <= 0.0:
        return 0.0

    ratio = current_area / previous_area

    return ratio if ratio <= 1.0 else 1.0 / ratio