# Sign identities expire after this many seconds. Signs don't move, but
# the camera does, so a short gap bridges a few missed/occluded frames
# without letting a dead track get revived by a different sign that
# later appears in a similar spot.
SIGN_GAP_SECONDS = 2.0

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

# --- Cross-class matching ---
# Sign classes flicker a lot at distance (a "60" can read as "80"), so
# bridging a class change matters here - but it must still be the same
# physical sign, which means real box overlap rather than mere
# proximity. Slightly looser than the vehicle thresholds because signs
# are static, so their boxes overlap strongly frame to frame.
SIGN_CROSS_CLASS_MATCH_THRESHOLD = 0.60
SIGN_CROSS_CLASS_MIN_IOU = 0.45

# --- Noise floor ---
# Lower than the vehicle threshold: a genuine sign is small on screen
# long before it is readable, and dropping those frames entirely would
# lose the track. This only filters out the very smallest specks.
SIGN_MIN_AREA_RATIO = 0.00002
