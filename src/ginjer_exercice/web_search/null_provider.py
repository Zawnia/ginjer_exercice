"""Null web-search provider used until Step 5b is implemented."""

from __future__ import annotations

from .base import WebSearchProvider, WebSearchResult
from ..schemas.ad import Brand
from ..schemas.products import ProductClassification


class NullWebSearchProvider(WebSearchProvider):
    """Stub provider for future product-name web verification."""

    def verify_product_name(
        self,
        brand: Brand,
        suggested_name: str,
        classification: ProductClassification,
    ) -> WebSearchResult:
        """Return a negative verification result without external I/O.

        Args:
            brand: Brand currently being processed.
            suggested_name: Candidate name proposed by Step 5a.
            classification: Product classification used as verification context.

        Returns:
            A stable negative verification result.
        """
        return WebSearchResult(
            confirmed=False,
            verified_name=None,
            source_url=None,
            confidence=0.0,
        )
