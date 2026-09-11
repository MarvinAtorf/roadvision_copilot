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
    "of guessing. If you are not sure, answer with 'I don't know'."
)

MAX_HISTORY_MESSAGES = 20  # last N messages to include


def _build_context_block(context: list[dict]) -> str:
    """Turn retrieved StVO paragraphs into a text block for the system prompt."""
    parts = [f"[{c['paragraph']}]\n{c['text']}" for c in context]
    return "\n\n".join(parts)


def ask(history: list[dict], context: list[dict] | None = None) -> str:
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    messages: list[MessageParam] = [{"role": m["role"], "content": m["content"]} for m in trimmed]

    system_prompt = SYSTEM_PROMPT
    if context:
        system_prompt += (
            "\n\nRelevant excerpts from the StVO (German Road Traffic Regulations) "
            "that may help answer the question. Only use facts stated in these "
            "excerpts — do not add anything beyond what they say:\n\n"
            + _build_context_block(context)
        )

    response: Message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
        stream=False,
    )
    block = response.content[0]
    if isinstance(block, TextBlock):
        return block.text
    raise ValueError(f"Unexpected content block type: {type(block)}")
