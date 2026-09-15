# Traffic lights remain available for matching for the complete video
TRAFFIC_LIGHT_GAP_SECONDS = None

# Identity matching threshold
TRAFFIC_LIGHT_MATCH_THRESHOLD = 0.30

# Spatial matching threshold (fraction of the image diagonal)
TRAFFIC_LIGHT_NEAR_DISTANCE = 0.05

# Size similarity threshold
TRAFFIC_LIGHT_SIZE_SIMILARITY = 0.30

# EMA smoothing
TRAFFIC_LIGHT_EMA_ALPHA = 0.20

# Remove duplicate detections
DEDUP_IOU_THRESHOLD = 0.60

# Identity score weights (IoU, center distance, size similarity)
SCORE_WEIGHT_IOU = 0.50
SCORE_WEIGHT_DISTANCE = 0.30
SCORE_WEIGHT_SIZE = 0.20
