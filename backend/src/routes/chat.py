import logging

from fastapi import APIRouter, HTTPException, Request
from llm.client import ask
from llm.video_context import build_timeline_context_block, build_video_context_block
from rag.bussgeldkatalog_index import retrieve_bussgeldkatalog_context
from rag.index import retrieve_context
from schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("roadvision")

router = APIRouter()


@router.post("/chat")
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    history = [m.model_dump() for m in payload.messages]

    context = []
    bussgeld_context = []
    if history:
        last_user_message = history[-1]["content"]
        try:
            context = retrieve_context(request.app.state.chroma_client, last_user_message)
        except Exception:
            logger.exception("RAG retrieval failed, continuing without context")

        try:
            bussgeld_context = retrieve_bussgeldkatalog_context(
                request.app.state.chroma_client, last_user_message
            )
        except Exception:
            logger.exception("Bußgeldkatalog retrieval failed, continuing without it")

    video_context_parts = []

    try:
        summary_block = build_video_context_block()
        if summary_block:
            video_context_parts.append(summary_block)
    except Exception:
        logger.exception("Loading video context failed, continuing without it")

    if history:
        try:
            timeline_block = build_timeline_context_block(last_user_message)
            if timeline_block:
                video_context_parts.append(timeline_block)
        except Exception:
            logger.exception("Loading video timeline context failed, continuing without it")

    video_context = "\n\n".join(video_context_parts) if video_context_parts else None

    try:
        answer = ask(
            history,
            context=context,
            video_context=video_context,
            bussgeld_context=bussgeld_context,
        )
    except Exception as err:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail="LLM request failed") from err
    return ChatResponse(answer=answer)
