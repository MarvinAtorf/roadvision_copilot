import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# Import Pydantic schemas and mappings
from backend.src.schemas.gtsign220_stvo_mapping import (
    traffic_sign_classes,
)

# Project root: roadvision_copilot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model path setup
model_path = PROJECT_ROOT / "backend" / "model_weights" / "traffic_sign_model.pt"
model = YOLO(str(model_path))

# Input photo path setup
image_path = (
    PROJECT_ROOT
    / "backend"
    / "pipelines"
    / "videos"
    / "raw_photos"
    / "test_frame_1.png"
)

# Output directory setup
output_folder = (
    PROJECT_ROOT
    / "backend"
    / "pipelines"
    / "videos"
    / "processed_photo"
)
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / "test_frame_1_result.png"
json_output_path = output_folder / "test_frame_1_result.json"

# Run inference
results = model.predict(
    source=str(image_path),
    imgsz=1280,
    conf=0.01,
    save=False,
)

print("Detected signs:", len(results[0].boxes))

img = cv2.imread(str(image_path))
detected_sign_models = []

# Map Class IDs to Pydantic objects for quick lookup
pydantic_mapping = {sign.gtsign_class_id: sign for sign in traffic_sign_classes}

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    confidence = float(box.conf[0])

    # Fetch sign metadata using the Pydantic schema
    sign_info = pydantic_mapping.get(cls_id)

    if sign_info:
        german_name = sign_info.german_official_name
        stvo_code = sign_info.stvo_code
        # Convert Pydantic object to dictionary for JSON output
        detected_sign_models.append({
            "confidence": round(confidence, 3),
            "sign_details": sign_info.model_dump()
        })
    else:
        german_name = model.names[cls_id]
        stvo_code = "Unknown"

    print(
        f"Class ID: {cls_id} ({stvo_code}), "
        f"German Name: {german_name}, "
        f"Confidence: {confidence:.3f}"
    )

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    label = f"{stvo_code} {confidence:.2f}"

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3,
    )

    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

cv2.imwrite(str(output_path), img)
print(f"Saved result image: {output_path}")

# Export Pydantic-validated dataset to JSON
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(detected_sign_models, f, indent=4, ensure_ascii=False)

print(f"Saved Pydantic-validated JSON: {json_output_path}")