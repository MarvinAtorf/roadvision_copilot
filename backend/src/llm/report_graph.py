import base64
import io
import operator
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from llm.client import client
from llm.video_context import load_video_analysis
from pipelines.shared.frame_export import extract_frame_as_jpeg
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "temp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ReportState(TypedDict):
    """Shared state passed between all nodes of the report graph."""

    analysis: dict | None
    output_video_path: str | None
    requested_moments: list[dict]  # each: {"timestamp_seconds": float, "note": str}
    frames: dict[float, bytes]
    # Written concurrently by parallel describe_single_frame branches, so it
    # needs a reducer telling LangGraph how to merge their partial updates
    # (dict union) instead of one overwriting another.
    descriptions: Annotated[dict[float, str], operator.or_]
    report_path: str | None
    error: str | None


def load_analysis(state: ReportState) -> ReportState:
    """Load the latest video analysis result from disk into the graph state."""
    result = load_video_analysis()

    if result is None or result.get("status") == "cancelled":
        state["error"] = "No completed video analysis available."
        return state

    state["analysis"] = result["data"]
    state["output_video_path"] = result.get("output_video")
    return state


def validate_moments(state: ReportState) -> ReportState:
    """
    Filter requested moments to those within the video's duration.

    Runs even if load_analysis already set an error, so it doesn't crash
    on a missing state["analysis"] — it just passes the error through
    unchanged, and later nodes are expected to do the same.
    """
    if state.get("error"):
        return state

    duration = state["analysis"]["video_metadata"]["duration_seconds"]

    valid_moments = [
        moment
        for moment in state["requested_moments"]
        if 0 <= moment["timestamp_seconds"] <= duration
    ]

    if not valid_moments:
        state["error"] = "None of the requested timestamps fall within the video's duration."

    state["requested_moments"] = valid_moments
    return state


def extract_frames(state: ReportState) -> ReportState:
    """Pull a JPEG frame for each validated moment from the processed video."""
    if state.get("error"):
        return state

    video_path = state["output_video_path"]
    if not video_path or not Path(video_path).exists():
        state["error"] = "Processed video file not found."
        return state

    fps = state["analysis"]["video_metadata"]["fps"]
    frames: dict[float, bytes] = {}

    for moment in state["requested_moments"]:
        timestamp = moment["timestamp_seconds"]
        frame_bytes = extract_frame_as_jpeg(video_path, timestamp, fps)
        if frame_bytes is not None:
            frames[timestamp] = frame_bytes

    if not frames:
        state["error"] = "Could not extract any frames for the requested moments."

    state["frames"] = frames
    return state


def dispatch_frame_descriptions(state: ReportState) -> list[Send] | str:
    """
    Fan out one describe_single_frame call per extracted frame, run in parallel.

    This is a conditional edge function, not a regular node: returning a
    list of Send objects tells LangGraph to run "describe_single_frame"
    once per frame, each with its own small input, all in parallel. If an
    earlier node already set an error, skip straight to assemble_report
    (returning a plain node-name string routes there normally instead).
    """
    if state.get("error"):
        return "assemble_report"

    notes_by_timestamp = {
        moment["timestamp_seconds"]: moment["note"] for moment in state["requested_moments"]
    }

    return [
        Send(
            "describe_single_frame",
            {
                "timestamp_seconds": timestamp,
                "frame_bytes": frame_bytes,
                "note": notes_by_timestamp.get(timestamp, ""),
            },
        )
        for timestamp, frame_bytes in state["frames"].items()
    ]


def describe_single_frame(payload: dict) -> dict:
    """
    Describe one frame via a vision-capable Claude call.

    Runs as a parallel branch dispatched by dispatch_frame_descriptions, so
    it receives only {"timestamp_seconds", "frame_bytes", "note"} — not the
    full ReportState. Its return value is a PARTIAL state update; the
    "descriptions" entries from every parallel branch are merged into the
    shared state via the operator.or_ reducer declared on that field.
    """
    timestamp = payload["timestamp_seconds"]
    note = payload["note"]
    image_b64 = base64.standard_b64encode(payload["frame_bytes"]).decode("utf-8")

    prompt = (
        f"This frame is from a traffic-analysis video, at approximately {timestamp:.0f} seconds. "
    )
    if note:
        prompt += f'The user flagged this moment with the note: "{note}". '
    prompt += (
        "Write a short description in natural, flowing prose — 3 to 5 sentences, "
        "no headings, no bullet points, no markdown formatting (no #, no **), and "
        "no technical details like track IDs, confidence scores, or frame numbers. "
        "Describe in plain language what's on the road: roughly how many and what "
        "kind of vehicles, whether a traffic light or specific traffic signs are "
        "visible, and the general situation. "
        "Do not read out or mention license plate numbers, street-facing house "
        "numbers, or any other personally identifiable text visible in the image, "
        "even partially — describe vehicles and buildings generically instead. "
        "If something looks potentially noteworthy (e.g. a vehicle appears to be "
        "in the intersection while a light looks red), you may mention it as an "
        "observation — never state it as a confirmed traffic violation, since a "
        "single still frame cannot establish that."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        description = response.content[0].text
    except Exception as exc:
        description = f"Description unavailable ({exc})."

    return {"descriptions": {timestamp: description}}


def _format_counts(counts: dict[str, int]) -> str:
    """Format a class->count dict as readable 'label: count' pairs, sorted
    by count descending and skipping any class with zero detections."""
    non_zero = {label: count for label, count in counts.items() if count > 0}
    sorted_items = sorted(non_zero.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{label}: {count}" for label, count in sorted_items)


def assemble_report(state: ReportState) -> ReportState:
    """Render the executive summary and per-moment sections into a PDF."""
    if state.get("error"):
        return state

    analysis = state["analysis"]
    metadata = analysis["video_metadata"]
    vehicles = analysis["vehicle_analysis"]
    lights = analysis["traffic_light_analysis"]
    signs = analysis["traffic_sign_analysis"]

    output_path = REPORTS_DIR / f"report_{uuid.uuid4().hex}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=22)
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#3d6db5"),
    )
    moment_heading_style = ParagraphStyle(
        "MomentHeading",
        parent=heading_style,
        fontSize=12,
    )
    body_style = styles["BodyText"]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    story = [
        Paragraph("RoadVision Copilot — Traffic Analysis Report", title_style),
        Spacer(1, 0.5 * cm),
        Paragraph("Summary", heading_style),
        Paragraph(f"Video duration: {metadata['duration_seconds']:.1f} s", body_style),
        Paragraph(f"Total unique vehicles: {vehicles['total_unique_vehicles']}", body_style),
        Paragraph(_format_counts(vehicles["vehicle_counts"]), body_style),
        Spacer(1, 0.15 * cm),
        Paragraph(f"Traffic lights tracked: {lights['total_tracked_traffic_lights']}", body_style),
        Spacer(1, 0.15 * cm),
        Paragraph(f"Unique traffic signs: {signs['total_unique_signs']}", body_style),
        Paragraph(_format_counts(signs["sign_counts"]), body_style),
        Spacer(1, 0.4 * cm),
        HRFlowable(width="100%", color=colors.lightgrey),
        Spacer(1, 0.4 * cm),
        Paragraph("Selected Moments", heading_style),
        Spacer(1, 0.2 * cm),
    ]

    video_width = metadata["width"]
    video_height = metadata["height"]
    image_width = 12 * cm
    image_height = image_width * (video_height / video_width)

    for timestamp in sorted(state["frames"].keys()):
        frame_bytes = state["frames"][timestamp]
        description = state["descriptions"].get(timestamp, "No description available.")

        moment_block = [
            Paragraph(f"At {timestamp:.0f}s", moment_heading_style),
            Spacer(1, 0.15 * cm),
            Image(io.BytesIO(frame_bytes), width=image_width, height=image_height),
            Spacer(1, 0.15 * cm),
            Paragraph(description, body_style),
            Spacer(1, 0.4 * cm),
        ]
        # Keeps image+description together on one page instead of splitting
        # across a page break mid-moment.
        story.append(KeepTogether(moment_block))

    doc.build(story)

    state["report_path"] = str(output_path)
    return state


_graph_builder = StateGraph(ReportState)

_graph_builder.add_node("load_analysis", load_analysis)
_graph_builder.add_node("validate_moments", validate_moments)
_graph_builder.add_node("extract_frames", extract_frames)
_graph_builder.add_node("describe_single_frame", describe_single_frame)
_graph_builder.add_node("assemble_report", assemble_report)

_graph_builder.add_edge(START, "load_analysis")
_graph_builder.add_edge("load_analysis", "validate_moments")
_graph_builder.add_edge("validate_moments", "extract_frames")

# extract_frames doesn't go to a fixed next node — dispatch_frame_descriptions
# decides at runtime: either fan out N parallel describe_single_frame calls
# (via Send) or, if something already failed, skip straight to assemble_report.

_graph_builder.add_conditional_edges(
    "extract_frames",
    dispatch_frame_descriptions,
    ["describe_single_frame", "assemble_report"],
)


# Every parallel describe_single_frame branch feeds back into the same
# next step once all of them finish.

_graph_builder.add_edge("describe_single_frame", "assemble_report")
_graph_builder.add_edge("assemble_report", END)

_report_graph = _graph_builder.compile()
