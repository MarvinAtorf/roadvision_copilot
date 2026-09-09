
from pydantic import BaseModel, Field


class TrafficSignDetection(BaseModel):
    gtsign_class_id: int = Field(..., description="GTSign220 class ID / GTSign220 Klassen-ID")
    stvo_code: str = Field(
        ..., description="Official StVO code (e.g., Sign 274-50) / Offizieller StVO-Code"
    )
    german_official_name: str = Field(
        ...,
        description="Official German name of the sign / Offizieller deutscher Name des Verkehrszeichens",
    )
    english_name: str = Field(
        ..., description="English name of the sign / Englischer Name des Verkehrszeichens"
    )
    confidence: float = Field(
        ..., description="Detection confidence score (0.0 - 1.0) / Erkennungsgenauigkeit"
    )
    bounding_box: list[float] | None = Field(
        None,
        description="Bounding box coordinates [x_min, y_min, x_max, y_max] / Begrenzungsrahmen",
    )


class VehicleMetrics(BaseModel):
    car_count: int = Field(
        0, description="Number of passenger cars / Anzahl der Personenkraftwagen (PKW)"
    )
    truck_count: int = Field(
        0, description="Number of trucks or heavy vehicles / Anzahl der Lastkraftwagen (LKW)"
    )
    bus_count: int = Field(0, description="Number of buses / Anzahl der Busse")
    motorcycle_count: int = Field(
        0, description="Number of motorcycles or two-wheelers / Anzahl der Motorräder"
    )
    total_vehicles: int = Field(
        0, description="Total number of vehicles / Gesamtzahl der Fahrzeuge"
    )


class TimelineFrame(BaseModel):
    timestamp_seconds: float = Field(
        ..., description="Timestamp of the frame in seconds / Zeitstempel des Frames in Sekunden"
    )
    frame_number: int = Field(
        ..., description="Corresponding video frame index / Entsprechende Video-Frame-Nummer"
    )
    traffic_density: str = Field(
        ...,
        description="Traffic density state (Low, Moderate, High, Gridlock) / Verkehrsdichte (Niedrig, Mäßig, Hoch, Stau)",
    )
    vehicles: VehicleMetrics = Field(
        ..., description="Vehicle counts for the specific frame / Fahrzeuganzahl für diesen Frame"
    )
    detected_signs: list[TrafficSignDetection] = Field(
        default_factory=list,
        description="Traffic signs detected in this frame / In diesem Frame erkannte Verkehrszeichen",
    )
    event_description: str | None = Field(
        None,
        description="Description of the event occurring at this timestamp in the user's preferred language (EN/DE) / Beschreibung des Ereignisses in der vom Benutzer bevorzugten Sprache",
    )


# analyzer betiğindeki import uyumluluğu için alias
TimelineEvent = TimelineFrame


class VideoTimelineAnalysis(BaseModel):
    video_id: str = Field(
        ...,
        description="Unique identifier or file name of the video / Eindeutige ID oder Dateiname des Videos",
    )
    duration_seconds: float = Field(
        ...,
        description="Total duration of the video in seconds / Gesamtdauer des Videos in Sekunden",
    )
    fps: float = Field(
        ..., description="Frame rate of the video (FPS) / Einzelbildrate des Videos (FPS)"
    )
    total_unique_signs_detected: int = Field(
        ...,
        description="Total count of unique traffic signs identified / Gesamtzahl eindeutig erkannten Verkehrszeichen",
    )
    summary: str = Field(
        ...,
        description="High-level narrative summary of the traffic situation in the user's preferred language (EN/DE) / Übersichtliche Zusammenfassung der Verkehrslage in der vom Benutzer bevorzugten Sprache",
    )
    timeline: list[TimelineFrame] = Field(
        ...,
        description="Sequential frame-by-frame or event-based timeline analysis / Sequenzielle Timeline-Analyse",
    )
