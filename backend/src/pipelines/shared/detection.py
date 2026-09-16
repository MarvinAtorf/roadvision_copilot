from pipelines.shared.config import TRAFFIC_LIGHT_CLASS, VEHICLE_CLASSES
from pipelines.shared.geometry import box_area


def extract_detections(result):
    """Turn a single YOLO result into a list of detection dicts.

    Returns both vehicle and traffic light detections mixed together;
    callers split them by class_id before handing them to the
    class-specific analyzer.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes

    # One host transfer per tensor instead of one per detection.
    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    detections = []

    for index in range(len(xyxy)):
        class_id = int(class_ids[index])

        class_name = VEHICLE_CLASSES.get(class_id)
        if class_name is None:
            if class_id != TRAFFIC_LIGHT_CLASS:
                continue
            class_name = "traffic_light"

        x1, y1, x2, y2 = xyxy[index]
        box = [float(x1), float(y1), float(x2), float(y2)]

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidences[index]),
                "box": box,
                "area": box_area(box),
            }
        )

    return detections


def extract_sign_detections(result):
    """Turn a single YOLO result from the sign-detection model into a
    list of detection dicts.

    Unlike extract_detections(), this does not filter against a fixed
    class map — the sign model's classes come straight from its own
    checkpoint (result.names), since it was trained on the GTSDB / sign
    taxonomy rather than COCO. class_id is kept as "gtsign class id" so
    the report/chat backend can look up the StVO meaning for each sign.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes
    names = result.names  # {class_id: class_name}, baked into the checkpoint

    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    detections = []

    for index in range(len(xyxy)):
        class_id = int(class_ids[index])
        class_name = names.get(class_id, f"sign_class_{class_id}")

        x1, y1, x2, y2 = xyxy[index]
        box = [float(x1), float(y1), float(x2), float(y2)]

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidences[index]),
                "box": box,
                "area": box_area(box),
            }
        )

    return detections
