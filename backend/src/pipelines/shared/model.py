from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[3] / "model_weights" / "yolo11m.pt"

_model: YOLO | None = None


def get_model() -> YOLO:
    """
    Retrieve a globally cached YOLO model instance. If the model is not
    already cached, it will attempt to load the model from a predefined
    path. If the model path does not exist, it raises a FileNotFoundError.

    Raises:
        FileNotFoundError: If the YOLO model cannot be found at the specified path.

    :return: A YOLO model instance.
    :rtype: YOLO
    """

    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found:\n{MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))

    return _model
