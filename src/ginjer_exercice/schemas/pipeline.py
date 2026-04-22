"""Schémas Pydantic pour les entrées/sorties du pipeline.

``PipelineOutput`` est le modèle canonique persisté par le ``ResultsRepository``.
La propriété ``needs_review`` est calculée à partir des produits détectés
et utilisée pour filtrer les résultats nécessitant une vérification humaine.
"""

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
    """Résultat final du pipeline pour une publicité.

    Attributes:
        ad_id: ID de la publicité traitée (platform_ad_id).
        brand: La marque de la publicité étudiée.
        products: Liste des produits détectés et classifiés.
        scores: Rapport des différents scores de la trace.
        trace_id: ID de la trace générée dans Langfuse.
    """
    ad_id: str = Field(..., description="ID de la publicité traitée (platform_ad_id)")
    brand: Brand = Field(..., description="La marque de la publicité étudiée")
    products: list[FinalProductLabel] = Field(
        default_factory=list,
        description="Liste des produits détectés",
    )
    scores: ScoreReport = Field(..., description="Rapport des différents scores de la trace")
    trace_id: str = Field(..., description="ID de la trace générée dans Langfuse")

    @property
    def needs_review(self) -> bool:
        """Indique si au moins un produit détecté nécessite une relecture humaine.

        Un résultat nécessite une review quand :
        - un produit a ``name_info.needs_review == True``
        - un produit n'a pas de ``name_info`` (nommage échoué)
        """
        return any(
            p.name_info is None or p.name_info.needs_review
            for p in self.products
        )
