from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock

client = Anthropic()

SYSTEM_PROMPT = (
    "You are an expert traffic analysis assistant for the RoadVision Copilot system. "
    "You answer questions about a traffic video that has been analyzed: detected traffic "
    "signs and their meaning under the German StVO (Road Traffic Regulations), as well as "
    "detected vehicles (e.g. counts, types) in the video. "
    "Answer only questions related to this traffic analysis context. "
    "Respond in English or German, depending on the language in which the question was asked. "
    "Be precise and factual. "
    "If the relevant data has not been provided to you, say so honestly instead of guessing. "
    "If you are not sure, answer with 'I don't know'."
)

MAX_HISTORY_MESSAGES = 20  # letzte N Nachrichten mitschicken


def ask(history: list[dict]) -> str:
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    messages: list[MessageParam] = [{"role": m["role"], "content": m["content"]} for m in trimmed]
    response: Message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
        stream=False,
    )
    block = response.content[0]
    if isinstance(block, TextBlock):
        return block.text
    raise ValueError(f"Unexpected content block type: {type(block)}")
