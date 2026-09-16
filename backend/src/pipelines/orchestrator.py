import json
from pathlib import Path

import cv2
from pipelines.live_status import update_status
from pipelines.shared.annotation import draw_dashboard, draw_detections
from pipelines.shared.config import (
    CONFIDENCE_THRESHOLD,
    FRAME_STRIDE,
    INFERENCE_BATCH_SIZE,
    INFERENCE_IMAGE_SIZE,
    SIGN_CONFIDENCE_THRESHOLD,
    SIGN_FRAME_STRIDE,
    SIGN_INFERENCE_IMAGE_SIZE,
    TRACKED_CLASSES,
)
from pipelines.shared.detection import extract_detections, extract_sign_detections
from pipelines.shared.encoding import reencode_to_h264
from pipelines.shared.model import get_model, get_sign_model
from pipelines.sign_analyzer.pipeline import SignAnalyzer
from pipelines.traffic_light_analyzer.pipeline import TrafficLightAnalyzer
from pipelines.vehicle_analyzer.pipeline import VehicleAnalyzer


def process_video(
    input_path: Path,
    output_path: Path,
    output_json_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run detection, tracking and annotation over a video file.

    Reads the video once and runs two batched YOLO passes per chunk of
    frames - one on the COCO-pretrained vehicle/traffic-light model
    (every FRAME_STRIDE-th frame), one on the sign-detection model
    (every SIGN_FRAME_STRIDE-th frame - signs stay on screen for a
    while, so they don't need re-detecting on every single frame the
    way fast-moving vehicles do). Publishes live progress via
    pipelines.live_status after every frame for the /status endpoint.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found:\n{input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = get_model()
    sign_model = get_sign_model()

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video:\n{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS detected.")

    diagonal = (width * width + height * height) ** 0.5

    if verbose:
        print("\nVideo information:")
        print(f"  Resolution: {width} x {height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {total_frames_in_video}")
        print(f"  Duration: {total_frames_in_video / fps:.2f} sec")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video:\n{output_path}")

    vehicle_analyzer = VehicleAnalyzer()
    traffic_light_analyzer = TrafficLightAnalyzer()
    sign_analyzer = SignAnalyzer()

    processed_frames = 0
    timeline = []
    last_detections: list[dict] = []
    last_sign_detections: list[dict] = []

    try:
        while True:
            frames = []
            for _ in range(INFERENCE_BATCH_SIZE):
                success, frame = cap.read()
                if not success:
                    break
                frames.append(frame)

            if not frames:
                break

            frame_indices = list(range(len(frames)))

            if FRAME_STRIDE > 1:
                inference_indices = [
                    i for i in frame_indices if (processed_frames + i) % FRAME_STRIDE == 0
                ]
            else:
                inference_indices = frame_indices

            if SIGN_FRAME_STRIDE > 1:
                sign_inference_indices = [
                    i for i in frame_indices if (processed_frames + i) % SIGN_FRAME_STRIDE == 0
                ]
            else:
                sign_inference_indices = frame_indices

            batch_results = {}
            batch_sign_results = {}

            if inference_indices:
                results = model(
                    [frames[i] for i in inference_indices],
                    classes=TRACKED_CLASSES,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=INFERENCE_IMAGE_SIZE,
                    verbose=False,
                )
                for position, frame_index in enumerate(inference_indices):
                    batch_results[frame_index] = results[position]

            if sign_inference_indices:
                sign_results = sign_model(
                    [frames[i] for i in sign_inference_indices],
                    conf=SIGN_CONFIDENCE_THRESHOLD,
                    imgsz=SIGN_INFERENCE_IMAGE_SIZE,
                    verbose=False,
                )
                for position, frame_index in enumerate(sign_inference_indices):
                    batch_sign_results[frame_index] = sign_results[position]

            for frame_index, frame in enumerate(frames):
                processed_frames += 1
                timestamp_seconds = (processed_frames - 1) / fps

                if frame_index in batch_results:
                    raw_detections = extract_detections(batch_results[frame_index])
                    last_detections = raw_detections
                else:
                    # Reuse the previous result for skipped frames.
                    raw_detections = last_detections

                if frame_index in batch_sign_results:
                    raw_sign_detections = extract_sign_detections(batch_sign_results[frame_index])
                    last_sign_detections = raw_sign_detections
                else:
                    raw_sign_detections = last_sign_detections

                vehicles = vehicle_analyzer.process_frame(
                    raw_detections, timestamp_seconds, diagonal
                )
                traffic_lights = traffic_light_analyzer.process_frame(
                    raw_detections, timestamp_seconds, diagonal
                )
                signs = sign_analyzer.process_frame(
                    raw_sign_detections, timestamp_seconds, diagonal
                )

                # Publish the current cumulative state for the /status endpoint.
                # traffic_light_count mirrors the same "max_visible_simultaneously"
                # metric used in the final report, not a cumulative unique count.
                vehicle_summary = vehicle_analyzer.build_summary()
                traffic_summary = traffic_light_analyzer.build_summary()
                sign_summary = sign_analyzer.build_summary()

                update_status(
                    frame_number=processed_frames,
                    total_frames_in_video=total_frames_in_video,
                    timestamp_seconds=round(timestamp_seconds, 3),
                    vehicle_counts=dict(vehicle_summary["vehicle_counts"]),
                    total_unique_vehicles=vehicle_summary["total_unique_vehicles"],
                    traffic_light_count=traffic_summary.get("max_visible_simultaneously", 0),
                    sign_counts=dict(sign_summary["sign_counts"]),
                    total_unique_signs=sign_summary["total_unique_signs"],
                )

                timeline.append(
                    {
                        "timestamp_seconds": round(timestamp_seconds, 3),
                        "frame_number": processed_frames,
                        "traffic_density": {"active_vehicles": len(vehicles)},
                        "vehicles": {
                            "active_count": len(vehicles),
                            "detections": [
                                {
                                    "track_id": int(d["canonical_id"]),
                                    "class": d["class_name"],
                                    "confidence": round(d["confidence"], 4),
                                    "bbox": [round(v, 2) for v in d["box"]],
                                }
                                for d in vehicles
                            ],
                        },
                        "traffic_lights": {
                            "visible_count": len(traffic_lights),
                            "detections": [
                                {
                                    "track_id": int(d["canonical_id"]),
                                    "confidence": round(d["confidence"], 4),
                                    "bbox": [round(v, 2) for v in d["box"]],
                                }
                                for d in traffic_lights
                            ],
                        },
                        "traffic_signs": {
                            "active_count": len(signs),
                            "detections": [
                                {
                                    "track_id": int(d["canonical_id"]),
                                    "class": d["class_name"],
                                    "sign_class_id": int(d["class_id"]),
                                    "confidence": round(d["confidence"], 4),
                                    "bbox": [round(v, 2) for v in d["box"]],
                                }
                                for d in signs
                            ],
                        },
                    }
                )

                annotated_signs = [{**d, "category": "traffic_sign"} for d in signs]
                draw_detections(frame, vehicles + traffic_lights + annotated_signs)

                draw_dashboard(
                    frame,
                    width,
                    [
                        f"Frame: {processed_frames}/{total_frames_in_video}",
                        f"Time: {timestamp_seconds:.2f}s",
                        f"Active vehicles: {len(vehicles)}",
                        f"Unique vehicles: {vehicle_summary['total_unique_vehicles']}",
                        f"Cars: {vehicle_summary['vehicle_counts']['car']}",
                        f"Trucks: {vehicle_summary['vehicle_counts']['truck']}",
                        f"Buses: {vehicle_summary['vehicle_counts']['bus']}",
                        f"Motorbikes: {vehicle_summary['vehicle_counts']['motorbike']}",
                        f"Bicycles: {vehicle_summary['vehicle_counts']['bicycle']}",
                        f"Traffic lights: {len(traffic_lights)}",
                        f"Traffic signs: {len(signs)}",
                    ],
                )

                writer.write(frame)

            if verbose and total_frames_in_video > 0:
                progress = processed_frames / total_frames_in_video * 100
                print(
                    f"\rProcessing: {processed_frames}/{total_frames_in_video} ({progress:.1f}%)",
                    end="",
                )

    finally:
        cap.release()
        writer.release()

    if verbose:
        print()

    analysis = {
        "video_metadata": {
            "video_path": str(input_path),
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "total_frames_in_video": total_frames_in_video,
            "frames_processed": processed_frames,
            "duration_seconds": round(processed_frames / fps, 3),
        },
        "vehicle_analysis": vehicle_analyzer.build_summary(),
        "traffic_light_analysis": traffic_light_analyzer.build_summary(),
        "traffic_sign_analysis": sign_analyzer.build_summary(),
        "timeline": timeline,
    }

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with output_json_path.open("w", encoding="utf-8") as json_file:
            json.dump(analysis, json_file, indent=2, ensure_ascii=False)

    return analysis


def run_video_analysis(input_video_path: str, output_video_path: str) -> dict:
    """Public entry point: run the full pipeline and re-encode to H.264."""
    input_path = Path(input_video_path)
    output_path = Path(output_video_path)

    results = process_video(
        input_path=input_path,
        output_path=output_path,
        output_json_path=output_path.with_suffix(".json"),
        verbose=False,
    )

    reencode_to_h264(output_path)

    return {
        "status": "success",
        "input_video": str(input_path),
        "output_video": str(output_path),
        "data": results,
    }