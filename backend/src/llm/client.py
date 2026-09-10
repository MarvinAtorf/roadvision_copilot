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
    "If the relevant data has not been provided to you, say so honestly instead of guessing. "
    "If you are not sure, answer with 'I don't know'."
)

MAX_HISTORY_MESSAGES = 20  # last N messages to include


def _build_context_block(context: list[dict]) -> str:
    """
    Constructs a context block from a list of dictionaries, where each dictionary
    contains a paragraph identifier and corresponding text. Each resulting block
    is formatted with the paragraph identifier enclosed in square brackets
    followed by the associated text.

    :param context: List of dictionaries where each dictionary must contain
        the keys 'paragraph' (str) and 'text' (str).
    :type context: list[dict]
    :return: A single formatted string where each context element is separated
        by two new lines.
    :rtype: str
    """
    parts = [f"[{c['paragraph']}]\n{c['text']}" for c in context]
    return "\n\n".join(parts)


def ask(history: list[dict], context: list[dict] | None = None) -> str:
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    messages: list[MessageParam] = [{"role": m["role"], "content": m["content"]} for m in trimmed]

    system_prompt = SYSTEM_PROMPT
    if context:
        system_prompt += (
            "\n\nRelevant excerpts from the StVO (German Road Traffic Regulations) "
            "that may help answer the question:\n\n" + _build_context_block(context)
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
