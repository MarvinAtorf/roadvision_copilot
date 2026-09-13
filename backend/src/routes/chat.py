import logging

from fastapi import APIRouter, HTTPException, Request
from llm.client import ask
from llm.video_context import build_video_context_block
from rag.index import retrieve_context
from schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("roadvision")

router = APIRouter()


@router.post("/chat")
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    history = [m.model_dump() for m in payload.messages]

    context = []
    if history:
        last_user_message = history[-1]["content"]
        try:
            context = retrieve_context(request.app.state.chroma_client, last_user_message)
        except Exception:
            logger.exception("RAG retrieval failed, continuing without context")

    video_context = None
    try:
        video_context = build_video_context_block()
    except Exception:
        logger.exception("Loading video context failed, continuing without it")

    try:
        answer = ask(history, context=context, video_context=video_context)
    except Exception as err:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail="LLM request failed") from err
    return ChatResponse(answer=answer)
