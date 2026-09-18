from pydantic import BaseModel, Field, model_validator


class ReportRangeRequest(BaseModel):
    """Request body for POST /analyze/video/report — a time range to summarize."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    scenario: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> "ReportRangeRequest":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self
