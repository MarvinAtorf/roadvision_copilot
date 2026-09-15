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
