"""
Shared helper for pulling a single frame out of a processed video as JPEG.

Used by both the report graph (extracting frames for user-selected
moments) and, later, the on-demand /analyze/video/frame endpoint.
"""

import cv2


def extract_frame_as_jpeg(video_path: str, timestamp_seconds: float, fps: float) -> bytes | None:
    frame_number = round(timestamp_seconds * fps)

    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            return None

        success, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if success else None
    finally:
        cap.release()
