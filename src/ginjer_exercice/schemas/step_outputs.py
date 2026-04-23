"""Schémas Pydantic pour les sorties intermédiaires de chaque step du pipeline.

Ces schémas sont distincts des schémas finaux de ``schemas/products.py`` :
ils représentent ce que le LLM renvoie directement, avant la transformation
métier éventuelle (post-validation, conversion de type, etc.).

Convention de nommage :
    - ``<StepN>LLMOutput``  → ce que le LLM génère (parsé par response_model)
    - ``UniverseResult``    → sortie publique de step1 (peut être identique au LLM output)
    - ``DetectedProductList`` → wrapper step2 (évite top-level array)
    - ``ProductClassification`` → déjà dans products.py, réexporté ici pour lisibilité
    - ``ExtractedName``     → sortie interne step4 avant conversion en ProductName
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Step 1 — Universe detection
# ──────────────────────────────────────────────────────────────


class UniverseDetection(BaseModel):
    """Un univers détecté avec son niveau de confiance et la justification."""

    universe: str = Field(
        ...,
        description="Nom de l'univers détecté (ex: Women, Beauty, Fragrance).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Niveau de confiance de la détection (0 = incertain, 1 = certain).",
    )
    reasoning: str = Field(
        ...,
        description="Justification brève appuyée sur un élément textuel ou visuel concret.",
    )


class UniverseResult(BaseModel):
    """Sortie de l'Étape 1 — liste des univers détectés dans la pub.

    Un univers de confiance < 0.4 peut être écarté en post-traitement.
    Le champ ``primary_universe`` est le premier univers par confiance décroissante.
    """

    detected_universes: list[UniverseDetection] = Field(
        default_factory=list,
        description="Liste ordonnée des univers détectés (du plus au moins probable).",
    )

    @property
    def primary_universe(self) -> str | None:
        """Retourne l'univers avec la plus haute confiance, ou None si liste vide."""
        if not self.detected_universes:
            return None
        return max(self.detected_universes, key=lambda u: u.confidence).universe

    @property
    def universe_names(self) -> list[str]:
        """Retourne la liste des noms d'univers pour les variables de prompt."""
        return [u.universe for u in self.detected_universes]


# ──────────────────────────────────────────────────────────────
# Step 2 — Product detection
# ──────────────────────────────────────────────────────────────


class DetectedProductLLM(BaseModel):
    """Représentation d'un produit telle que renvoyée par le LLM à l'étape 2.

    Note : ``color`` est une string libre ici — la validation vers l'enum ``Color``
    se fait en post-traitement dans step2_products.py.
    """

    raw_description: str = Field(
        ...,
        description="Description précise et spécifique du produit (matière, forme, détails visuels).",
    )
    universe: str = Field(
        ...,
        description="L'univers auquel appartient ce produit.",
    )
    color: str = Field(
        ...,
        description="Couleur dominante du produit (ex: Black, Beige, Gold).",
    )
    importance: int = Field(
        ...,
        ge=0,
        le=5,
        description="Importance visuelle du produit dans la pub (0=marginal, 5=sujet principal).",
    )


class DetectedProductList(BaseModel):
    """Wrapper de la liste de produits détectés.

    Encapsuler dans un objet est plus robuste qu'un top-level array
    pour Gemini et OpenAI, et permet d'ajouter des métadonnées globales.
    """

    products: list[DetectedProductLLM] = Field(
        default_factory=list,
        description="Liste des produits détectés. Vide si aucun produit identifiable.",
    )
    overall_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confiance globale du LLM dans sa détection de produits.",
    )


# ──────────────────────────────────────────────────────────────
# Step 3 — Taxonomy classification (LLM output)
# Note : ProductClassification (le schéma final) est dans schemas/products.py.
# On l'utilise directement comme response_model car il correspond exactement
# à ce que le LLM doit renvoyer.
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# Step 4 — Name extraction
# ──────────────────────────────────────────────────────────────


class ExtractedName(BaseModel):
    """Sortie interne du LLM pour l'extraction de nom (step 4).

    Convertie en ``ProductName`` par ``step4_name.execute()`` avant
    d'être renvoyée à l'orchestrator.
    """

    name: str | None = Field(
        default=None,
        description="Nom explicite du produit tel qu'il apparaît dans le contenu. null si absent.",
    )
    found_in: Literal["body_text", "title", "caption", "url", "image_text", "none"] = Field(
        ...,
        description="Source où le nom a été trouvé.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiance dans l'extraction (0 = très incertain, 1 = certain).",
    )
