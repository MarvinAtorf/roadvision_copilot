import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Import Pydantic models
from backend.src.schemas.gtsign220_stvo_mapping import traffic_sign_classes
from backend.src.schemas.video_timeline import VideoTimelineAnalysis, TimelineEvent

# Project root: roadvision_copilot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input video
VIDEO_PATH = (
    PROJECT_ROOT
    / "backend"
    / "pipelines"
    / "videos"
    / "raw_videos"
    / "Bildschirmaufnahme_youtube_1.mov"
)

# Models
VEHICLE_MODEL_PATH = PROJECT_ROOT / "backend" / "model_weights" / "yolo11m.pt"
SIGN_MODEL_PATH = PROJECT_ROOT / "backend" / "model_weights" / "traffic_sign_model.pt"

# Outputs
OUTPUT_VIDEO_PATH = (
    PROJECT_ROOT
    / "backend"
    / "pipelines"
    / "videos"
    / "processed_videos"
    / "output_counted.mp4"
)

OUTPUT_JSON_PATH = (
    PROJECT_ROOT
    / "backend"
    / "pipelines"
    / "videos"
    / "processed_videos"
    / "traffic_analysis.json"
)

# Load Models
vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
sign_model = YOLO(str(SIGN_MODEL_PATH))

# Map Class IDs to Pydantic models
pydantic_mapping = {sign.gtsign_class_id: sign for sign in traffic_sign_classes}

# Moving vehicles to track
VEHICLE_CLASSES = [1, 2, 3, 4, 5, 7]

# Separate class for static infrastructure tracking
TRAFFIC_LIGHT_CLASS = [9]

CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "minibus",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 165, 0),
    5: (255, 0, 255),
    7: (255, 255, 0),
}

TRAFFIC_LIGHT_COLOR = (0, 165, 255)

def overlap_fraction(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return intersection / area_a

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / max(1, union)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise Exception(f"Video cannot be opened: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

scale_factor = height / 720.0
dash_rect_x1, dash_rect_y1 = int(10 * scale_factor), int(10 * scale_factor)
dash_rect_x2, dash_rect_y2 = int(300 * scale_factor), int(250 * scale_factor)
dash_text_x_start, dash_text_y_start = int(20 * scale_factor), int(35 * scale_factor)

# Independent tracking memories
counted_vehicle_ids = set()
counted_tl_ids = set()

counts_by_class = {c: 0 for c in VEHICLE_CLASSES}
rider_ids = set()

OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

timeline_events = []
processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frames += 1
    current_time_sec = round(processed_frames / fps, 2)

    # 1. Track Vehicles (Class IDs: 1, 2, 3, 4, 5, 7)
    vehicle_results = vehicle_model.track(
        frame, persist=True, tracker="bytetrack.yaml", classes=VEHICLE_CLASSES, verbose=False
    )

    # 2. Track Traffic Lights as a separate class pipeline (Class ID: 9)
    tl_results = vehicle_model.track(
        frame, persist=True, tracker="bytetrack.yaml", classes=TRAFFIC_LIGHT_CLASS, verbose=False
    )
    
    # 3. Detect Traffic Signs
    sign_results = sign_model(frame, verbose=False)

    # --- PROCESS TRAFFIC SIGNS ---
    detected_signs = []
    if sign_results[0].boxes is not None:
        sign_boxes = sign_results[0].boxes.xyxy.cpu().numpy()
        sign_classes = sign_results[0].boxes.cls.int().cpu().numpy()

        for box, cls in zip(sign_boxes, sign_classes):
            cls_id = int(cls)
            sign_info = pydantic_mapping.get(cls_id)

            sign_payload = {
                "class_id": cls_id,
                "box": box.tolist(),
                "details": sign_info.model_dump() if sign_info else {"raw_label": sign_model.names[cls_id]}
            }
            detected_signs.append(sign_payload)

            x1, y1, x2, y2 = box
            label_text = sign_info.stvo_code if sign_info else f"Sign {cls_id}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- PROCESS TRACKED TRAFFIC LIGHTS ---
    tracked_traffic_lights = []
    if tl_results[0].boxes is not None and tl_results[0].boxes.id is not None:
        tl_boxes = tl_results[0].boxes.xyxy.cpu().numpy()
        tl_ids = tl_results[0].boxes.id.int().cpu().numpy()

        for i in range(len(tl_ids)):
            tl_id = int(tl_ids[i])
            box = tl_boxes[i]

            counted_tl_ids.add(tl_id)
            tracked_traffic_lights.append({
                "traffic_light_id": tl_id,
                "box": box.tolist()
            })

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), TRAFFIC_LIGHT_COLOR, 2)
            cv2.putText(
                frame,
                f"Traffic Light #{tl_id}",
                (int(x1), max(10, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TRAFFIC_LIGHT_COLOR,
                1,
            )

    # --- PROCESS TRACKED VEHICLES ---
    active_vehicle_counts = {c: 0 for c in VEHICLE_CLASSES}
    raw_detections = []

    if vehicle_results[0].boxes is not None and vehicle_results[0].boxes.id is not None:
        boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
        ids = vehicle_results[0].boxes.id.int().cpu().numpy()
        classes = vehicle_results[0].boxes.cls.int().cpu().numpy()

        for i in range(len(ids)):
            raw_detections.append((int(ids[i]), int(classes[i]), boxes[i]))

    filtered_detections = []
    raw_detections.sort(key=lambda x: (x[0] not in counted_vehicle_ids, x[0]))

    for tid, cid, box in raw_detections:
        is_duplicate = False
        for f_tid, f_cid, f_box in filtered_detections:
            if cid == f_cid and (iou(box, f_box) > 0.55 or overlap_fraction(box, f_box) > 0.6):
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_detections.append((tid, cid, box))

    current_frame_detections = filtered_detections
    bike_boxes_in_frame = [box for tid, cid, box in current_frame_detections if cid in [1, 3]]

    for tid, cid, box in current_frame_detections:
        if cid == 0:
            for b_box in bike_boxes_in_frame:
                if overlap_fraction(box, b_box) > 0.25:
                    rider_ids.add(tid)

    for tid, cid, box in current_frame_detections:
        if cid == 0 and tid in rider_ids:
            continue

        if tid not in counted_vehicle_ids:
            counted_vehicle_ids.add(tid)
            if cid in counts_by_class:
                counts_by_class[cid] += 1

        if cid in active_vehicle_counts:
            active_vehicle_counts[cid] += 1

        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(cid, (255, 255, 255))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if cid in CLASS_NAMES:
            cv2.putText(
                frame,
                f"{CLASS_NAMES[cid]} #{tid}",
                (int(x1), max(10, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    active_total = sum(active_vehicle_counts.values())
    if active_total < 5:
        density_status = "Low (Free Flow)"
    elif active_total <= 10:
        density_status = "Medium (Moderate)"
    else:
        density_status = "High (Congested)"

    # Build Pydantic timeline event object
    try:
        event = TimelineEvent(
            timestamp=current_time_sec,
            frame=processed_frames,
            vehicles=counts_by_class.copy(),
            traffic_signs=detected_signs,
            traffic_lights=tracked_traffic_lights,
            density=density_status,
        )
        timeline_events.append(event)
    except Exception:
        timeline_events.append({
            "timestamp": current_time_sec,
            "frame": processed_frames,
            "vehicles": counts_by_class.copy(),
            "traffic_signs": detected_signs,
            "traffic_lights": tracked_traffic_lights,
            "density": density_status,
        })

    # Render dashboard overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (dash_rect_x1, dash_rect_y1), (dash_rect_x2, dash_rect_y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    out.write(frame)

cap.release()
out.release()

# Save structured analysis
try:
    analysis_result = VideoTimelineAnalysis(
        video_name="Bildschirmaufnahme_youtube_1.mov",
        duration_seconds=duration,
        total_frames=processed_frames,
        events=timeline_events
    )
    final_json_data = analysis_result.model_dump()
except Exception:
    final_json_data = [e.model_dump() if hasattr(e, "model_dump") else e for e in timeline_events]

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(final_json_data, f, indent=4, ensure_ascii=False)

print(f"Analysis saved to: {OUTPUT_JSON_PATH}")