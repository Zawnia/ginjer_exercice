"""Schemas Pydantic pour les entrees/sorties du pipeline.

``PipelineOutput`` est le modele canonique persiste par le ``ResultsRepository``.
La propriete ``needs_review`` est calculee a partir des produits detectes
et utilisee pour filtrer les resultats necessitant une verification humaine.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from .ad import Brand
from .products import FinalProductLabel
from .scores import ScoreReport


class PipelineInput(BaseModel):
    ad_id: str = Field(..., description="ID de la publicite a traiter")


class StepResult(BaseModel):
    step_name: str
    success: bool
    data: Any | None = None
    error: str | None = None


class PipelineOutput(BaseModel):
    """Resultat final du pipeline pour une publicite."""

    ad_id: str = Field(..., description="ID de la publicite traitee (platform_ad_id)")
    brand: Brand = Field(..., description="La marque de la publicite etudiee")
    products: list[FinalProductLabel] = Field(
        default_factory=list,
        description="Liste des produits detectes",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings non bloquants collectes pendant l'execution du pipeline",
    )
    scores: ScoreReport = Field(..., description="Rapport des differents scores de la trace")
    trace_id: str = Field(..., description="ID de la trace generee dans Langfuse")

    @computed_field(return_type=Literal["clean", "degraded"])
    @property
    def quality_status(self) -> Literal["clean", "degraded"]:
        """Synthese de qualite derivee des warnings runtime."""
        return "degraded" if self.warnings else "clean"

    @property
    def needs_review(self) -> bool:
        """Indique si au moins un produit detecte necessite une relecture humaine."""
        return any(
            p.name_info is None or p.name_info.needs_review
            for p in self.products
        )
