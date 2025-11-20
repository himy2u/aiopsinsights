"""Data models for the bronze ingestion pipeline."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


class Gender(str, Enum):
    """Patient gender options."""
    MALE = "M"
    FEMALE = "F"
    NON_BINARY = "X"
    UNKNOWN = "U"


class InputChannel(str, Enum):
    """Input channel types."""
    WEB = "web"
    SMS = "sms"
    PHONE = "phone"


class BronzeRecord(BaseModel):
    """Raw input record in bronze layer."""
    id: str = Field(..., description="Unique record identifier")
    timestamp: datetime = Field(..., description="Record creation timestamp")
    channel: InputChannel = Field(..., description="Input channel")
    patient_text: str = Field(..., description="Raw patient symptom description")
    zip_code: Optional[str] = Field(
        None, min_length=5, max_length=10, description="Patient ZIP code"
    )
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age in years")
    gender: Gender = Field(
        default=Gender.UNKNOWN, description="Patient gender"
    )

    @validator("zip_code")
    def validate_zip_code(cls, v):
        if v is not None and not v.isdigit() and "-" not in v:
            raise ValueError("ZIP code must be numeric or in 12345-6789 format")
        return v


class SilverRecord(BaseModel):
    """Normalized record in silver layer."""
    id: str = Field(..., description="Unique record identifier")
    timestamp: datetime = Field(..., description="Record creation timestamp")
    source: InputChannel = Field(..., description="Input channel")
    symptom_text: str = Field(..., description="PII-redacted symptom text")
    demographics: dict = Field(
        default_factory=dict,
        description="Anonymized demographic information"
    )
    pii_redacted: bool = Field(
        default=False, description="Whether PII has been redacted"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2023-11-19T14:30:00Z",
                "source": "web",
                "symptom_text": "I have a sharp pain in my [REDACTED] when I breathe deep and my left arm feels [REDACTED].",
                "demographics": {
                    "age": 45,
                    "gender": "U"
                },
                "pii_redacted": True
            }
        }
