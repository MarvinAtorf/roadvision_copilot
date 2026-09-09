import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse

from backend.src.pipelines.video_traffic_analyzer import run_video_analysis

router = APIRouter()


@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)): # noqa: B008
    # 1. Save the incoming video file to a temporary directory
    temp_dir = Path(tempfile.gettempdir())
    raw_path = temp_dir / f"raw_{file.filename}"
    output_path = temp_dir / f"processed_{file.filename}"

    with raw_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run the video analysis pipeline
    run_video_analysis(str(raw_path), str(output_path))

    # 3. Return the processed video back to the client
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"processed_{file.filename}",
    )
