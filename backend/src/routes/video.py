import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from llm.report_graph import _report_graph
from pipelines.analysis_lock import finish, try_start
from pipelines.cancel_flag import clear_cancel, request_cancel
from pipelines.live_status import get_status, reset_status
from pipelines.orchestrator import run_video_analysis
from schemas.report import ReportRangeRequest

router = APIRouter()

# Latest video analysis JSON (overwritten on every new analysis)
TEMP_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "temp"
TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
LATEST_ANALYSIS_JSON = TEMP_DATA_DIR / "roadvision_latest_analysis.json"


@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):  # noqa: B008
    # Only one analysis may run at a time (MVP2 scope) — reject immediately,
    # before touching disk, if another analysis already holds the slot.
    if not try_start():
        raise HTTPException(status_code=409, detail="An analysis is already in progress")

    try:
        temp_dir = Path(tempfile.gettempdir())

        raw_path = temp_dir / f"raw_{file.filename}"
        output_path = temp_dir / f"processed_{file.filename}"

        with raw_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        reset_status()
        clear_cancel()

        # Run the blocking analysis in a worker thread so the event loop stays
        # free to serve GET /analyze/video/status while this request is in flight.
        analysis_result = await run_in_threadpool(
            run_video_analysis,
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

        if analysis_result["status"] == "cancelled":
            return JSONResponse(
                status_code=200,
                content={"status": "cancelled"},
            )

        # Keep the existing MVP-1 video response unchanged
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"processed_{file.filename}",
        )
    finally:
        finish()


@router.post("/analyze/video/cancel")
async def cancel_video_analysis():
    """Signal the running analysis to stop as soon as possible."""
    request_cancel()
    return {"status": "cancel_requested"}


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


@router.post("/analyze/video/report")
async def generate_report(payload: ReportRangeRequest, request: Request):
    """Generate a PDF report summarizing a user-selected time range of the video,
    optionally filtered to frames matching a user-described scenario."""
    initial_state = {
        "analysis": None,
        "output_video_path": None,
        "chroma_client": request.app.state.chroma_client,
        "time_range": {
            "start_seconds": payload.start_seconds,
            "end_seconds": payload.end_seconds,
        },
        "scenario": payload.scenario,
        "requested_moments": [],
        "frames": {},
        "descriptions": {},
        "scenario_matches": {},
        "report_path": None,
        "error": None,
    }

    # Runs the graph (frame sampling + extraction + parallel vision calls +
    # PDF rendering) in a worker thread, same reasoning as analyze_video:
    # keeps the event loop free while this blocking work happens.
    result = await run_in_threadpool(_report_graph.invoke, initial_state)

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    return FileResponse(
        path=result["report_path"],
        media_type="application/pdf",
        filename="roadvision_report.pdf",
    )
