from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    service_status = request.app.state.service_status
    return {
        "api": "ok",
        "chroma_db": "ok" if service_status["chromadb"] == "ok" else "degraded",
    }
