from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Valid irrigation sources accepted by the engine
IrrigationSource = Literal["rain_fed", "canal", "borewell", "drip", "sprinkler"]


class PredictionInput(BaseModel):
    nitrogen: Optional[float] = Field(default=None, ge=0)
    phosphorous: Optional[float] = Field(default=None, ge=0)
    potassium: Optional[float] = Field(default=None, ge=0)
    ph: Optional[float] = Field(default=None, ge=0)
    temperature_c: Optional[float] = None
    humidity: Optional[float] = Field(default=None, ge=0)
    rainfall_mm: Optional[float] = Field(default=None, ge=0)
    moisture: Optional[float] = Field(default=None, ge=0)
    area: Optional[float] = Field(default=None, gt=0)
    price_per_ton: Optional[float] = Field(default=None, ge=0)
    season: Optional[str] = None
    state_name: Optional[str] = None
    district_name: Optional[str] = None
    latitude: Optional[float] = Field(default=None, ge=6.0, le=38.5)
    longitude: Optional[float] = Field(default=None, ge=68.0, le=98.0)
    crop_year: Optional[int] = Field(default=None, ge=1990, le=2100)
    top_k: int = Field(default=3, ge=1, le=10)
    # New context fields
    previous_crop: Optional[str] = Field(
        default=None,
        description="Crop grown last season. Used to penalise mono-cropping.",
    )
    irrigation_source: Optional[IrrigationSource] = Field(
        default=None,
        description="Primary water source. Rain-fed farms get penalty for high-water crops.",
    )
    soil_type: Optional[str] = Field(
        default=None,
        description="Soil type preset (alluvial / black_cotton / red_laterite / sandy_loam / clay).",
    )


class TrainRequest(BaseModel):
    data_dir: Optional[str] = None
