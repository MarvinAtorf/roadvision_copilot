import re

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock

client = Anthropic()

SYSTEM_PROMPT = (
    "You are an expert traffic analysis assistant for the RoadVision Copilot system. "
    "You answer two kinds of questions: "
    "(1) questions about a specific traffic video that has been analyzed, including "
    "detected traffic signs and their meaning under the German StVO (Road Traffic "
    "Regulations), and detected vehicles (e.g. counts, types) in that video, and "
    "(2) general questions about German traffic law (the StVO) itself. "
    "Answer only questions related to these two topics; decline anything unrelated. "
    "Respond in English or German, depending on the language in which the question was asked. "
    "Be precise and factual. "
    "CRITICAL — never invent facts: only state specific details (dates, numbers, "
    "paragraph references, sign numbers, deadlines, exceptions) that are explicitly "
    "present in the provided context or conversation. Do not add plausible-sounding "
    "additional examples, dates, or details from general knowledge to make an answer "
    "seem more complete. If the provided context only partially answers the question, "
    "answer with only what the context supports and explicitly say which part you "
    "cannot confirm — do not fill the gap with unconfirmed information. "
    "If the relevant data has not been provided to you at all, say so honestly instead "
    "of guessing. If you are not sure, answer with 'I don't know'. "
    "If asked what happens at a specific point in time in the video, and a block "
    "tagged [video_timeline_<seconds>s] is provided, describe only what that block "
    "states for its time window — do not imply frame-exact precision, since the data "
    "is aggregated over a ~20 second window around that point. If the block is tagged "
    "[video_timeline_out_of_range] instead, tell the user the requested time falls "
    "outside the video's duration rather than guessing at an answer. "
    "CITATIONS — every context block you are given is tagged with an exact source id "
    "in square brackets, e.g. [stvo_full_§3], [stvo_signs_274], or [video_timeline_80s]. "
    "Whenever you use a fact from a block, append its exact tag at the end of the "
    "relevant sentence — the citation format depends on the block's source, not a "
    "single fixed style: "
    "for StVO blocks (tags starting with stvo_full_ or stvo_signs_), use "
    "':blue[**(rag: [<tag>])**]'; "
    "for video-analysis blocks (tags starting with video_timeline_), use "
    "':violet[**(video_context: [<tag>])**]'. "
    "This is Streamlit-flavored markdown for colored, bold text; copy the tag itself "
    "exactly as given, never invent or reformat it. If your answer uses no provided "
    "context (e.g. you had to say 'I don't know'), add no citation. Never cite a tag "
    "that was not actually provided to you in this turn."
)

MAX_HISTORY_MESSAGES = 20  # last N messages to include

PARAGRAPH_NUMBER_PATTERN = re.compile(r"§\s*(\d+[a-z]?)")


def _paragraph_tag(paragraph_header: str) -> str:
    """Turn a full paragraph header (e.g. "§ 3 Geschwindigkeit") into a short,
    stable citation tag (e.g. "stvo_full_§3") for the model to cite verbatim."""
    match = PARAGRAPH_NUMBER_PATTERN.search(paragraph_header)
    number = match.group(1) if match else paragraph_header
    return f"**stvo_full_§{number}**"


def _build_context_block(context: list[dict]) -> str:
    """Turn retrieved StVO paragraphs into a tagged text block for the system prompt."""
    parts = [f"[{_paragraph_tag(c['paragraph'])}]\n{c['text']}" for c in context]
    return "\n\n".join(parts)


def ask(
    history: list[dict],
    context: list[dict] | None = None,
    video_context: str | None = None,
) -> str:
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    messages: list[MessageParam] = [{"role": m["role"], "content": m["content"]} for m in trimmed]

    system_prompt = SYSTEM_PROMPT

    if video_context:
        system_prompt += (
            "\n\nData from the most recently analyzed traffic video. Only use facts "
            "stated here — do not add anything beyond what is given. This may include "
            "a block about a specific point in time, tagged [video_timeline_<seconds>s] "
            "or [video_timeline_out_of_range] — cite it with the violet "
            "'video_context:' style described above:\n\n" + video_context
        )

    if context:
        system_prompt += (
            "\n\nRelevant excerpts from the StVO (German Road Traffic Regulations) "
            "that may help answer the question. Only use facts stated in these "
            "excerpts — do not add anything beyond what they say. Each excerpt is "
            "tagged with its exact citation id in square brackets:\n\n"
            + _build_context_block(context)
        )

    response: Message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        stream=False,
    )
    block = response.content[0]
    if isinstance(block, TextBlock):
        return block.text
    raise ValueError(f"Unexpected content block type: {type(block)}")
