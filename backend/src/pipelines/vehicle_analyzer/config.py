# Vehicle identities expire after this many seconds
VEHICLE_GAP_SECONDS = 6.0

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