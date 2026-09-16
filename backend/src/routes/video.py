import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pipelines.live_status import get_status, reset_status

from pipelines.orchestrator import run_video_analysis

router = APIRouter()

# Latest video analysis JSON (overwritten on every new analysis)
TEMP_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "temp"
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
LATEST_ANALYSIS_JSON = TEMP_DATA_DIR / "roadvision_latest_analysis.json"

# Processed videos are kept here instead of only in the OS temp dir, so a
# run can still be reviewed after the request is over (and, in Docker,
# after the container restarts - the temp dir does not survive that).
PROCESSED_VIDEO_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_videos"
PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):  # noqa: B008
    temp_dir = Path(tempfile.gettempdir())

    raw_path = temp_dir / f"raw_{file.filename}"
    output_path = temp_dir / f"processed_{file.filename}"

    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reset_status()

    # Run the blocking analysis in a worker thread so the event loop stays
    # free to serve GET /analyze/video/status while this request is in flight.
    analysis_result = await run_in_threadpool(
        run_video_analysis,
        str(raw_path),
        str(output_path),
    )

    # Keep a timestamped copy so previous runs aren't overwritten by the
    # next upload of a file with the same name.
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    stored_video_path = PROCESSED_VIDEO_DIR / f"{timestamp}_{file.filename}"
    shutil.copy2(output_path, stored_video_path)

    stored_json_path = stored_video_path.with_suffix(".json")

    analysis_result["stored_video"] = str(stored_video_path)

    with stored_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(analysis_result, json_file, indent=2, ensure_ascii=False)

    # Save latest analysis result for the frontend
    with LATEST_ANALYSIS_JSON.open("w", encoding="utf-8") as json_file:
        json.dump(
            analysis_result,
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    # Keep the existing MVP-1 video response unchanged
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"processed_{file.filename}",
    )


@router.get("/analyze/video/processed")
async def list_processed_videos():
    """List the processed videos kept on disk, newest first."""
    videos = sorted(
        PROCESSED_VIDEO_DIR.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return {
        "count": len(videos),
        "videos": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "created": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            for path in videos
        ],
    }


@router.get("/analyze/video/processed/{filename}")
async def get_processed_video(filename: str):
    """Return one previously processed video by filename."""
    # Resolve and confirm the path stays inside PROCESSED_VIDEO_DIR, so a
    # crafted filename can't read arbitrary files off the server.
    candidate = (PROCESSED_VIDEO_DIR / filename).resolve()

    if (
        PROCESSED_VIDEO_DIR.resolve() not in candidate.parents
        or not candidate.is_file()
    ):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Video not found."},
        )

    return FileResponse(path=candidate, media_type="video/mp4", filename=candidate.name)


@router.get("/analyze/video/status")
async def get_video_status():
    """Return the live cumulative counts while a video is being analyzed."""
    status = get_status()

    if status is None:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "No analysis in progress.",
            },
        )

    return status


@router.get("/analyze/video/analysis")
async def get_video_analysis():
    """
    Return the analysis JSON from the latest processed video.
    """

    if not LATEST_ANALYSIS_JSON.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": "No video analysis available yet.",
            },
        )

    with LATEST_ANALYSIS_JSON.open("r", encoding="utf-8") as json_file:
        analysis = json.load(json_file)

    return analysis


@router.delete("/analyze/video/analysis")
async def clear_video_analysis():
    """
    Clear the server-side video analysis.
    """
    if LATEST_ANALYSIS_JSON.exists():
        LATEST_ANALYSIS_JSON.unlink()

    return {"status": "success"}
