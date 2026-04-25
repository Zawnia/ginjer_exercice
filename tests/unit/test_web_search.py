from __future__ import annotations

from ginjer_exercice.schemas.ad import Brand
from ginjer_exercice.schemas.products import ProductClassification
from ginjer_exercice.web_search.null_provider import NullWebSearchProvider


def test_null_web_search_provider_returns_negative_result() -> None:
    provider = NullWebSearchProvider()

    result = provider.verify_product_name(
        brand=Brand.CHANEL,
        suggested_name="N°5",
        classification=ProductClassification(
            universe="Beauty",
            category="Perfume",
            subcategory="Women's Perfume",
            confidence=0.9,
        ),
    )

    assert result.confirmed is False
    assert result.verified_name is None
    assert result.source_url is None
    assert result.confidence == 0.0
