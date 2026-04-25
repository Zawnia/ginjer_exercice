"""Abstract web-search interfaces for Step 5 verification."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ..schemas.ad import Brand
from ..schemas.products import ProductClassification


class WebSearchResult(BaseModel):
    """Outcome of a product-name verification attempt."""

    confirmed: bool = Field(..., description="Whether the product name was confirmed.")
    verified_name: str | None = Field(default=None, description="Verified product name if confirmed.")
    source_url: str | None = Field(default=None, description="Supporting source URL if available.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the verification result.")


class WebSearchProvider(ABC):
    """Verify that a suggested product name exists on official brand sources."""

    @abstractmethod
    def verify_product_name(
        self,
        brand: Brand,
        suggested_name: str,
        classification: ProductClassification,
    ) -> WebSearchResult:
        """Verify a suggested product name.

        Args:
            brand: Brand currently being processed.
            suggested_name: Candidate product name proposed by Step 5a.
            classification: Product classification used as verification context.

        Returns:
            Verification result for the suggested name.
        """
