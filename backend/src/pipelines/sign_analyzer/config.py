# Sign identities expire after this many seconds. Signs don't move, but
# the camera does, so a short gap (similar to traffic lights) is enough
# to bridge a few missed/occluded frames without merging two different
# signs that happen to appear in the same spot later.
SIGN_GAP_SECONDS = 4.0

# Identity matching thresholds
SIGN_MATCH_THRESHOLD = 0.40

# Spatial matching threshold (fraction of the image diagonal)
SIGN_NEAR_DISTANCE = 0.03

# Size similarity threshold
SIGN_SIZE_SIMILARITY = 0.40

# EMA smoothing
SIGN_EMA_ALPHA = 0.60

# Remove duplicate detections
DEDUP_IOU_THRESHOLD = 0.60

# Identity score weights (IoU, center distance, size similarity)
SCORE_WEIGHT_IOU = 0.50
SCORE_WEIGHT_DISTANCE = 0.30
SCORE_WEIGHT_SIZE = 0.20
