from fastapi import APIRouter
from schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat")
def chat(payload: ChatRequest) -> ChatResponse:
    return ChatResponse(answer="Hello World")
