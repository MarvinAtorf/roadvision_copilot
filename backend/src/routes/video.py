from fastapi import APIRouter

router = APIRouter()


@router.post("/analyze/video")
def analyze_video():
    return {"message": "Video analysis endpoint"}
