import base64
import io
import operator
import re
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import chromadb
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from llm.client import client
from llm.video_context import load_video_analysis
from pipelines.shared.frame_export import extract_frame_as_jpeg
from rag.bussgeldkatalog_index import retrieve_bussgeldkatalog_context
from rag.sign_index import get_sign_by_number
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from traffic_sign_mapping.sign_class_mapping import lookup_by_class_name

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "temp" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Caps how many frames a single time-range request can produce, regardless
# of how long the range is — keeps the number of vision calls bounded.
MAX_RANGE_MOMENTS = 12
MIN_MOMENT_SPACING_SECONDS = 1.0

# Caps how many distinct signs get a lookup per frame, so a bucket with
# many detected sign types doesn't balloon the prompt.
MAX_SIGNS_PER_FRAME = 5

_MATCH_LINE_PATTERN = re.compile(r"^\s*MATCH:\s*(yes|no)\s*$", re.IGNORECASE)


class ReportState(TypedDict):
    """Shared state passed between all nodes of the report graph."""

    analysis: dict | None
    output_video_path: str | None
    chroma_client: chromadb.ClientAPI
    time_range: dict | None  # {"start_seconds": float, "end_seconds": float}
    scenario: str | None
    requested_moments: list[dict]  # derived internally: [{"timestamp_seconds", "note"}]
    frames: dict[float, bytes]
    # Written concurrently by parallel describe_single_frame branches, so
    # both need a reducer telling LangGraph how to merge their partial
    # updates (dict union) instead of one overwriting another.
    descriptions: Annotated[dict[float, str], operator.or_]
    scenario_matches: Annotated[dict[float, bool], operator.or_]
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


def derive_moments_from_range(state: ReportState) -> ReportState:
    """
    Turn a user-given time range into a capped list of evenly spaced moments.

    Sampling instead of describing every frame keeps the number of vision
    calls bounded regardless of how long the requested range is. Both
    start and end are clamped into [0, duration] — a start beyond the
    video's length must not survive past this point, or a later, larger
    end-clamp alone could leave start > end.
    """
    if state.get("error"):
        return state

    duration = state["analysis"]["video_metadata"]["duration_seconds"]

    start = min(max(state["time_range"]["start_seconds"], 0.0), duration)
    end = min(max(state["time_range"]["end_seconds"], 0.0), duration)

    if end <= start:
        state["error"] = "The requested time range does not overlap with the video's duration."
        return state

    span = end - start
    interval = max(span / MAX_RANGE_MOMENTS, MIN_MOMENT_SPACING_SECONDS)

    moments = []
    timestamp = start
    while timestamp <= end and len(moments) < MAX_RANGE_MOMENTS:
        moments.append({"timestamp_seconds": round(timestamp, 1), "note": ""})
        timestamp += interval

    state["requested_moments"] = moments
    return state


def validate_moments(state: ReportState) -> ReportState:
    """
    Filter requested moments to those within the video's duration.

    Acts as a defense-in-depth check after derive_moments_from_range (which
    already clips against the duration) — cheap and harmless to keep.
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
        state["error"] = "None of the derived timestamps fall within the video's duration."

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


def _find_nearby_sign_names(analysis: dict, timestamp: float) -> list[str]:
    """
    Return the traffic-sign class names detected in the timeline_summary
    bucket nearest to this timestamp.

    Same nearest-bucket lookup as video_context.build_timeline_context_block
    — reused here by re-implementing the same small logic rather than
    importing from the chat module, since the two live in different layers
    (video_context.py is chat-prompt-specific).
    """
    summary = analysis.get("timeline_summary")
    if not summary:
        return []

    nearest_bucket = min(
        summary, key=lambda bucket: abs(bucket["bucket_center_seconds"] - timestamp)
    )
    return list(nearest_bucket.get("sign_counts", {}).keys())


def _build_bussgeld_context_block(chroma_client: chromadb.ClientAPI, scenario: str) -> str:
    """
    Look up the single most relevant Bußgeldkatalog category for the
    scenario description, once per report (not per frame — the query is
    the same scenario text regardless of which frame is being described).

    Returns an empty string if nothing relevant is found (e.g. RAG lookup
    fails, or the catalog isn't indexed) — callers must treat that as "no
    hypothetical fine information available", not an error.
    """
    try:
        matches = retrieve_bussgeldkatalog_context(chroma_client, scenario, top_k=1)
    except Exception:
        matches = []

    if not matches:
        return ""

    return (
        f"\n\nGeneral Bußgeldkatalog (German fine catalog) information for the category "
        f"'{matches[0]['category']}', for reference only:\n{matches[0]['text']}"
    )


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

    scenario = state.get("scenario")
    bussgeld_context_block = (
        _build_bussgeld_context_block(state["chroma_client"], scenario) if scenario else ""
    )

    return [
        Send(
            "describe_single_frame",
            {
                "timestamp_seconds": timestamp,
                "frame_bytes": frame_bytes,
                "note": notes_by_timestamp.get(timestamp, ""),
                "scenario": scenario,
                "bussgeld_context": bussgeld_context_block,
                "nearby_signs": _find_nearby_sign_names(state["analysis"], timestamp),
                "chroma_client": state["chroma_client"],
            },
        )
        for timestamp, frame_bytes in state["frames"].items()
    ]


def _build_sign_context_block(chroma_client: chromadb.ClientAPI, sign_names: list[str]) -> str:
    """
    Look up official StVO information for each nearby detected sign and
    format it as a plain-text block for the prompt — deduplicated and
    capped at MAX_SIGNS_PER_FRAME.

    Deterministic lookup: detected class name -> StVO sign number via the
    verified GTSDB->StVO mapping, then an exact-ID fetch from ChromaDB —
    no embedding similarity search involved. Similarity search on short
    English class names (e.g. "no entry") against the German-language
    catalog was returning unrelated signs (e.g. "no entry" -> Zeichen 341,
    "Wartelinie" — completely wrong), so this replaces that approach
    entirely for the signs the detector actually recognizes.
    """
    seen = set()
    lines = []

    for sign_name in sign_names:
        if sign_name in seen or len(lines) >= MAX_SIGNS_PER_FRAME:
            continue
        seen.add(sign_name)

        mapping = lookup_by_class_name(sign_name)
        if mapping is None:
            continue

        entry = get_sign_by_number(chroma_client, mapping.sign_number)
        if entry:
            lines.append(f"- {entry['text']} (Quelle: Zeichen {entry['sign_number']})")

    if not lines:
        return ""

    return (
        "\n\nOfficial StVO information about traffic signs detected near this "
        "moment, for reference:\n" + "\n".join(lines)
    )


def _extract_match_line(text: str, scenario_given: bool) -> tuple[str, bool]:
    """
    Split off a trailing 'MATCH: yes'/'MATCH: no' line from the model's
    response, used later to filter which frames make it into the report.

    Without a scenario, every frame counts as a match — there's nothing to
    filter, all sampled frames go into the report exactly as before. With
    a scenario, a missing or unparseable MATCH line defaults to no-match:
    safer to under-include than to silently include an unverified frame.
    """
    if not scenario_given:
        return text.strip(), True

    lines = text.strip().splitlines()
    if lines:
        match = _MATCH_LINE_PATTERN.match(lines[-1])
        if match:
            description = "\n".join(lines[:-1]).strip()
            return description, match.group(1).lower() == "yes"

    return text.strip(), False


_HYPOTHETICAL_FINE_PATTERN = re.compile(r"\bHypothetically,.*", re.IGNORECASE | re.DOTALL)


def _strip_unearned_fine_mention(description: str, matched: bool) -> str:
    """
    Remove a hypothetical-fine sentence the model attached despite answering
    MATCH: no.

    The prompt tells the model to add the fine sentence only when the
    relevant context is present (MATCH: yes), and to always start it with
    "Hypothetically, ...". Under ambiguous or hedged reasoning the model
    sometimes adds it anyway while still (correctly) answering MATCH: no —
    this is a deterministic backstop so a Bußgeldkatalog citation never ends
    up attached to a frame that wasn't actually confirmed as relevant.
    """
    if matched:
        return description
    return _HYPOTHETICAL_FINE_PATTERN.sub("", description).strip()


def describe_single_frame(payload: dict) -> dict:
    """
    Describe one frame via a vision-capable Claude call.

    Runs as a parallel branch dispatched by dispatch_frame_descriptions, so
    it receives only its own small payload — not the full ReportState. Its
    return value is a PARTIAL state update; entries from every parallel
    branch are merged into the shared state via the operator.or_ reducers
    declared on "descriptions" and "scenario_matches".
    """
    timestamp = payload["timestamp_seconds"]
    note = payload["note"]
    scenario = payload.get("scenario")
    bussgeld_context_block = payload.get("bussgeld_context", "")
    image_b64 = base64.standard_b64encode(payload["frame_bytes"]).decode("utf-8")

    sign_context_block = _build_sign_context_block(
        payload["chroma_client"], payload.get("nearby_signs", [])
    )

    prompt = (
        f"This frame is from a traffic-analysis video, at approximately {timestamp:.0f} seconds. "
    )
    if note:
        prompt += f'The user flagged this moment with the note: "{note}". '

    prompt += (
        "Describe in plain language, in natural flowing prose, roughly 3 to 5 "
        "sentences, what's visible on the road: roughly how many and what kind "
        "of vehicles, whether a traffic light or specific traffic signs are "
        "visible, and the general situation. Keep it concise — this length "
        "limit applies even when there is more below to consider. "
        "No headings, no bullet points, no markdown formatting (no #, no **), "
        "and no technical details like track IDs, confidence scores, or frame "
        "numbers. Do not read out or mention license plate numbers, "
        "street-facing house numbers, or any other personally identifiable "
        "text visible in the image, even partially — describe vehicles and "
        "buildings generically instead. Do not name specific vehicle makes or "
        "models (e.g. 'Audi', 'BMW') — describe vehicles by type and color "
        "instead (e.g. 'a dark SUV'), since make/model identification from a "
        "traffic camera frame isn't reliable enough to state as fact."
    )

    if scenario:
        prompt += (
            f"\n\nThe user is asking a hypothetical 'what if' question: "
            f'"{scenario}". This is NOT asking whether a vehicle actually did '
            "this in the video — it's asking what would apply IF this action or "
            "situation occurred here. Your job is to judge whether THIS FRAME "
            "shows the situational context relevant to that question — the "
            "elements needed to meaningfully answer it — not whether a vehicle "
            "is shown actually doing it. For example: for a 'drove through a red "
            "light' question, the relevant context is an intersection with a red "
            "light visible for this direction of travel — whether or not any "
            "vehicle is shown crossing it. For a right-of-way ('rechts vor "
            "links') question, the relevant context is an intersection with no "
            "traffic light or priority/stop sign, regardless of whether a "
            "vehicle is shown crossing it unsafely. For a question about a "
            "vehicle's position or spacing right now (e.g. riding close between "
            "two other vehicles), the relevant context is that spatial situation "
            "itself, since it's directly observable in a still image. "
            "If the scenario concerns something that requires observing motion "
            "or change over time to identify even the relevant context (e.g. "
            "vehicle speed, acceleration) — not just the hypothetical action "
            "itself — say so explicitly (e.g. 'a still frame cannot show vehicle "
            "speed') and answer 'MATCH: no', since guessing here would be "
            "misleading rather than helpful. "
            "If the relevant context IS present, describe what you see that "
            "establishes it, then address the hypothetical: what would apply if "
            "the described action or situation occurred here, still within the "
            "3-to-5-sentence limit above. Do not claim any vehicle in the frame "
            "actually did this — describe only what establishes the context, "
            "and frame the rest purely as the hypothetical answer."
        )

        if bussgeld_context_block:
            prompt += (
                "\n\nIf — and only if — the relevant context for the hypothetical "
                "is present, add one final sentence clearly framed as a "
                "hypothetical, starting with something like 'Hypothetically, if "
                "this action were taken here, ...', naming only the fine amount "
                "and points that the Bußgeldkatalog information below generally "
                "states for the matching case — cite it in plain text using "
                "exactly this format: (Quelle: Bußgeldkatalog - <category>). "
                "No markdown, no color tags, since this text goes directly into a "
                "PDF. Never imply any vehicle in the frame actually did this, and "
                "never invent a fine amount not present in the information "
                "below." + bussgeld_context_block
            )

        prompt += (
            "\n\nIf the relevant context is NOT present in this frame, write only "
            "one short sentence saying so — briefly note what IS visible instead, "
            "since this frame may still be shown in the report as evidence even "
            "when it does not match. "
            "After your description, on its own new line, write exactly "
            "'MATCH: yes' if this frame shows the situational context relevant "
            "to the hypothetical, or 'MATCH: no' if it does not. "
            "Important: 'MATCH: yes' means the CONTEXT needed to answer the "
            "hypothetical is visible — not that any vehicle actually performed "
            "the action. Only answer 'MATCH: no' when the relevant context is "
            "genuinely absent (e.g. no red light, no intersection, or no "
            "relevant sign visible for a rule-based question) or when the "
            "scenario itself requires seeing motion that a still image cannot "
            "show, as above. This line is required and must be the last line "
            "of your reply."
        )
    else:
        prompt += (
            "\n\nPay particular attention to any potential for an accident — for "
            "example vehicles following too closely, a vehicle or pedestrian on a "
            "possible collision path, someone stepping into the road, or an "
            "otherwise unsafe or confusing situation. Also note if the frame shows "
            "an unsignalized intersection (no traffic light, no priority or stop "
            "sign visible) with vehicles approaching from different directions, "
            "since this is where right-of-way ('right before left') applies by "
            "default in Germany. Only mention any of this if something in the "
            "image actually suggests it; do not invent a risk if the scene looks "
            "unremarkable, and keep such observations clearly tentative (e.g. "
            "'possibly', 'it's unclear whether') rather than stated as fact — and "
            "still keep the overall description within the 3-to-5-sentence limit "
            "above."
        )

    prompt += sign_context_block
    if sign_context_block:
        prompt += (
            "\n\nIf the official StVO information above is actually relevant to "
            "what you see in the image, you may briefly incorporate it into your "
            "description in plain language, citing it inline in plain text using "
            "exactly this format: (Quelle: Zeichen <number>) — no markdown, no "
            "color tags, since this text goes directly into a PDF. Only cite a "
            "sign number that was actually given to you above; never invent one."
        )

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=600,
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
        raw_text = response.content[0].text
    except Exception as exc:
        raw_text = f"Description unavailable ({exc})."

    description, matched = _extract_match_line(raw_text, scenario_given=bool(scenario))
    description = _strip_unearned_fine_mention(description, matched)

    return {
        "descriptions": {timestamp: description},
        "scenario_matches": {timestamp: matched},
    }


def _format_counts(counts: dict[str, int]) -> str:
    """Format a class->count dict as readable 'label: count' pairs, sorted
    by count descending and skipping any class with zero detections."""
    non_zero = {label: count for label, count in counts.items() if count > 0}
    sorted_items = sorted(non_zero.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{label}: {count}" for label, count in sorted_items)


def _build_moment_block(
    timestamp: float,
    frame_bytes: bytes,
    description: str,
    image_width: float,
    image_height: float,
    moment_heading_style: ParagraphStyle,
    body_style: ParagraphStyle,
) -> Flowable:
    """
    Build the image+description flowable for a single moment, kept together
    on one page.

    Factored out of assemble_report so the exact same rendering is used both
    for matching moments (scenario given, MATCH: yes) and, since a report
    should stay verifiable even when nothing matched, for the sampled
    moments shown as evidence when a scenario has zero matches.

    :param timestamp: The moment's timestamp in seconds.
    :param frame_bytes: JPEG bytes of the extracted frame.
    :param description: The model-written description for this moment.
    :param image_width: Rendered image width, in ReportLab units.
    :param image_height: Rendered image height, in ReportLab units.
    :param moment_heading_style: Paragraph style for the "At Ns" heading.
    :param body_style: Paragraph style for the description text.
    :return: A KeepTogether flowable ready to append to the report story.
    """
    return KeepTogether(
        [
            Paragraph(f"At {timestamp:.0f}s", moment_heading_style),
            Spacer(1, 0.15 * cm),
            Image(io.BytesIO(frame_bytes), width=image_width, height=image_height),
            Spacer(1, 0.15 * cm),
            Paragraph(description, body_style),
            Spacer(1, 0.4 * cm),
        ]
    )


def assemble_report(state: ReportState) -> ReportState:
    """Render the executive summary and per-moment sections into a PDF."""
    if state.get("error"):
        return state

    analysis = state["analysis"]
    metadata = analysis["video_metadata"]
    vehicles = analysis["vehicle_analysis"]
    lights = analysis["traffic_light_analysis"]
    signs = analysis["traffic_sign_analysis"]
    scenario = state.get("scenario")

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

    video_width = metadata["width"]
    video_height = metadata["height"]
    image_width = 12 * cm
    image_height = image_width * (video_height / video_width)

    # scenario_matches defaults every frame to True when no scenario was
    # given (see _extract_match_line), so this filter is a no-op in that
    # case and every sampled frame is included, same as before. Computed
    # here (before the story is built) so the top-of-report verdict banner
    # below can use it too.
    matching_timestamps = sorted(
        ts for ts in state["frames"] if state["scenario_matches"].get(ts, True)
    )

    story = [
        Paragraph("RoadVision Copilot — Traffic Analysis Report", title_style),
        Spacer(1, 0.4 * cm),
    ]

    # Scenario verdict goes first, right under the title — before the
    # summary — so a reader (or Marvin during the presentation) sees
    # immediately whether the tested case came back positive or negative,
    # without having to scroll or read the frame-by-frame descriptions.
    if scenario:
        matched_overall = bool(matching_timestamps)
        verdict_style = ParagraphStyle(
            "ScenarioVerdict",
            parent=styles["Heading1"],
            fontSize=15,
            textColor=colors.HexColor("#1a7a33") if matched_overall else colors.HexColor("#b3261e"),
            spaceBefore=2,
            spaceAfter=6,
        )
        verdict_text = (
            "✓ MATCH — evidence found" if matched_overall else "✗ NO MATCH — no evidence found"
        )

        story.append(Paragraph("Scenario Check", heading_style))
        story.append(Paragraph(f'Checked against: "{scenario}"', body_style))
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(verdict_text, verdict_style))
        story.append(Spacer(1, 0.4 * cm))
        story.append(HRFlowable(width="100%", color=colors.lightgrey))
        story.append(Spacer(1, 0.4 * cm))

    story += [
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
    ]

    if not scenario:
        story.append(Paragraph("Selected Moments", heading_style))
        story.append(Spacer(1, 0.2 * cm))
    elif matching_timestamps:
        story.append(Paragraph("Matching Moment", heading_style))
        story.append(Spacer(1, 0.2 * cm))
        if len(matching_timestamps) > 1:
            story.append(
                Paragraph(
                    f"{len(matching_timestamps)} of the sampled moments matched the "
                    "scenario; the earliest one is shown below as representative "
                    "evidence.",
                    body_style,
                )
            )
            story.append(Spacer(1, 0.2 * cm))

    if scenario and not matching_timestamps:
        story.append(
            Paragraph(
                f"In the {len(state['frames'])} frame(s) sampled between "
                f"{state['time_range']['start_seconds']:.0f}s and "
                f"{state['time_range']['end_seconds']:.0f}s, none showed evidence "
                f"matching the described scenario. This does not rule out that it "
                "occurred outside the sampled frames — only a sample of moments "
                "in this range was checked, not every frame.",
                body_style,
            )
        )
        story.append(Spacer(1, 0.3 * cm))

        # Show what was actually sampled, so a non-match stays verifiable
        # instead of a bare claim — the reviewer can see exactly what each
        # checked moment looked like and judge the "no match" for themselves.
        story.append(Paragraph("Sampled moments", moment_heading_style))
        story.append(Spacer(1, 0.2 * cm))

        for timestamp in sorted(state["frames"]):
            description = state["descriptions"].get(timestamp, "No description available.")
            story.append(
                _build_moment_block(
                    timestamp,
                    state["frames"][timestamp],
                    description,
                    image_width,
                    image_height,
                    moment_heading_style,
                    body_style,
                )
            )
    else:
        # Once the scenario is confirmed true in this video, one representative
        # frame is enough — showing every matching frame just repeats the same
        # finding. The earliest match is used since it's the first moment the
        # situation is established.
        for timestamp in matching_timestamps[:1]:
            description = state["descriptions"].get(timestamp, "No description available.")
            story.append(
                _build_moment_block(
                    timestamp,
                    state["frames"][timestamp],
                    description,
                    image_width,
                    image_height,
                    moment_heading_style,
                    body_style,
                )
            )

    doc.build(story)

    state["report_path"] = str(output_path)
    return state


_graph_builder = StateGraph(ReportState)

_graph_builder.add_node("load_analysis", load_analysis)
_graph_builder.add_node("derive_moments_from_range", derive_moments_from_range)
_graph_builder.add_node("validate_moments", validate_moments)
_graph_builder.add_node("extract_frames", extract_frames)
_graph_builder.add_node("describe_single_frame", describe_single_frame)
_graph_builder.add_node("assemble_report", assemble_report)

_graph_builder.add_edge(START, "load_analysis")
_graph_builder.add_edge("load_analysis", "derive_moments_from_range")
_graph_builder.add_edge("derive_moments_from_range", "validate_moments")
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
