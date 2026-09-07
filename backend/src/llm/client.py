from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlock

client = Anthropic()

SYSTEM_PROMPT = (
    "You are an expert on the German StVO (Road Traffic Regulations). "
    "Answer only questions about traffic signs and traffic rules of the StVO. "
    "Respond in English or German, depending on the language in which the "
    "question was asked. Be precise and factual. "
    "If you are not sure, answer with 'I don't know'."
    "NEVER use emoji's or Markdown"
)

MAX_HISTORY_MESSAGES = 10  # maximum last 10 messages


def ask(history: list[dict]) -> str:
    """
    Processes the conversation history and generates a response based on it using a pre-configured
    AI model. The function trims the history to the last N messages (defined by MAX_HISTORY_MESSAGES)
    to ensure efficient processing and then interacts with an AI client to produce a response.

    :param history: A list of dictionaries representing the conversation history where each dictionary
        contains information such as roles and contents of the conversation.
    :type history: list[dict]
    :return: The generated response from the AI model as a string.
    :rtype: str
    :raises ValueError: If the content returned from the model is not of the expected TextBlock type.
    """

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
