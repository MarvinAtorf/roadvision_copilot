VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}

TRAFFIC_LIGHT_CLASS = 9

TRACKED_CLASSES = [*VEHICLE_CLASSES.keys(), TRAFFIC_LIGHT_CLASS]

CONFIDENCE_THRESHOLD = 0.25

# Number of frames sent to YOLO in a single batch. Larger batches use
# the GPU far better; on CPU the gain is smaller but still measurable.
INFERENCE_BATCH_SIZE = 8


FRAME_STRIDE = 1

INFERENCE_IMAGE_SIZE = 640

# --- Traffic sign detection ---
# Separate GTSDB/gtsign-trained model (traffic_sign_model.pt), run as its
# own inference pass since its classes don't exist in the COCO-pretrained
# vehicle model. Signs are small in the frame, so we run inference at a
# higher resolution than the vehicle model (matches the imgsz used during
# training / in single_frame_sign_detector.py).
SIGN_CONFIDENCE_THRESHOLD = 0.25
SIGN_INFERENCE_IMAGE_SIZE = 1280

# Signs stay visible on screen for a while as the vehicle approaches, so
# re-running the (expensive, high-res) sign model on every single frame
# is wasted work. Only run it every SIGN_FRAME_STRIDE-th frame; skipped
# frames reuse the previous sign detections. This is independent from
# FRAME_STRIDE (which applies to the vehicle/traffic-light model).
SIGN_FRAME_STRIDE = 3