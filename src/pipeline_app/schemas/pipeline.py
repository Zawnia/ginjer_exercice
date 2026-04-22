from pydantic import BaseModel, Field
from typing import Any

from .ad import Brand
from .products import FinalProductLabel
from .scores import ScoreReport

class PipelineInput(BaseModel):
    ad_id: str = Field(..., description="ID de la publicité à traiter")

class StepResult(BaseModel):
    step_name: str
    success: bool
    data: Any | None = None
    error: str | None = None

class PipelineOutput(BaseModel):
    ad_id: str = Field(..., description="ID de la publicité traitée (platform_ad_id)")
    brand: Brand = Field(..., description="La marque de la publicité étudiée")
    products: list[FinalProductLabel] = Field(
        default_factory=list, 
        description="Liste des produits détectés"
    )
    scores: ScoreReport = Field(..., description="Rapport des différents scores de la trace")
    trace_id: str = Field(..., description="ID de la trace générée dans Langfuse")
