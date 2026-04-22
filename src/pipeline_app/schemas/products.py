from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal 


class Color(str, Enum):
    BLACK = "Black"
    WHITE = "White"
    NAVY = "Navy"
    BROWN = "Brown"
    BEIGE = "Beige"
    GREY = "Grey"
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    YELLOW = "Yellow"
    PURPLE = "Purple"
    PINK = "Pink"
    ORANGE = "Orange"
    GOLD = "Gold"
    SILVER = "Silver"
    MULTICOLOR = "Multicolor"

class DetectedProduct(BaseModel):
    """Sortie de l'Étape 2 : Identification visuelle et textuelle basique."""
    importance: int = Field(..., ge=0, le=5, description="Importance visuelle de 0 à 5.")
    color: Color
    universe: str = Field(..., description="L'univers global détecté (ex: Fashion, Beauty).")
    raw_description: str = Field(..., description="Description brute pour le fallback.")

class ProductClassification(BaseModel):
    """Sortie de l'Étape 3 : Classification dans l'arbre de la marque."""
    universe: str
    category: str
    subcategory: str
    product_type: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de l'IA entre 0 et 1.")

class ProductName(BaseModel):
    """Sortie de l'Étape 4 ou 5 : Le nom final du produit."""
    name: str | None = None
    source: Literal["explicit", "fallback_llm", "fallback_web"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_review: bool
    sources_consulted: list[str] = Field(default_factory=list)

class FinalProductLabel(BaseModel):
    """Assemble les résultats de toutes les étapes pour un produit donné."""
    detected: DetectedProduct
    classification: ProductClassification
    name_info: ProductName | None = None

    def __str__(self):
        color = self.detected.color.value
        subcat = self.classification.subcategory
        name = self.name_info.name if self.name_info and self.name_info.name else "Produit Inconnu"
        
        return f"{name} - {color} - {subcat}"