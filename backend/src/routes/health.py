from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    """
    Health check endpoint.
    """
    service_status = request.app.state.service_status
    return {
        "api": "ok",
        "chroma_db": service_status["chromadb"],
    }
