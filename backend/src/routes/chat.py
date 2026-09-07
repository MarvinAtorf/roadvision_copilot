import logging

from fastapi import APIRouter, HTTPException
from llm.client import ask
from schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("roadvision")

router = APIRouter()


@router.post("/chat")
def chat(payload: ChatRequest) -> ChatResponse:
    history = [m.model_dump() for m in payload.messages]
    try:
        answer = ask(history)
    except Exception as err:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail="LLM request failed") from err
    return ChatResponse(answer=answer)
