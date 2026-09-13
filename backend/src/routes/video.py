import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pipelines.video_traffic_analyzer import run_video_analysis

router = APIRouter()

# Latest video analysis JSON (overwritten on every new analysis)
TEMP_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "temp"
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
LATEST_ANALYSIS_JSON = TEMP_DATA_DIR / "roadvision_latest_analysis.json"


@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):  # noqa: B008
    temp_dir = Path(tempfile.gettempdir())

    raw_path = temp_dir / f"raw_{file.filename}"
    output_path = temp_dir / f"processed_{file.filename}"

    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run full video analysis
    analysis_result = run_video_analysis(
        str(raw_path),
        str(output_path),
    )

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
