# Vehicle identities expire after this many seconds.
# Kept short on purpose: traffic moves fast, and a long memory lets a
# dead track linger and get "revived" by a completely different vehicle
# that happens to pass through a similar position later.
VEHICLE_GAP_SECONDS = 1.5

# Identity matching thresholds
VEHICLE_MATCH_THRESHOLD = 0.40

# Spatial matching threshold (fraction of the image diagonal)
VEHICLE_NEAR_DISTANCE = 0.03

# Size similarity threshold
VEHICLE_SIZE_SIMILARITY = 0.40

# EMA smoothing
VEHICLE_EMA_ALPHA = 0.60

# Remove duplicate detections
DEDUP_IOU_THRESHOLD = 0.60

# Identity score weights (IoU, center distance, size similarity)
SCORE_WEIGHT_IOU = 0.50
SCORE_WEIGHT_DISTANCE = 0.30
SCORE_WEIGHT_SIZE = 0.20

# --- Cross-class matching ---
# Continuing a track across a class change exists ONLY to bridge
# classifier flicker on one physical vehicle (COCO models wobble between
# car/truck/bus on SUVs, vans and lorries). It must never let one track
# swallow a genuinely different vehicle, so a class change demands far
# stronger geometric evidence than a same-class continuation: a high
# overall score AND real box overlap.
VEHICLE_CROSS_CLASS_MATCH_THRESHOLD = 0.70
VEHICLE_CROSS_CLASS_MIN_IOU = 0.55

# --- Noise floor ---
# Detections smaller than this fraction of diagonal^2 are dropped before
# tracking. Very small boxes are where the model's class predictions are
# least reliable, and they produce most of the phantom tracks.
VEHICLE_MIN_AREA_RATIO = 0.0003
