"""Tests unitaires pour Step 2 — Product detection."""

from __future__ import annotations

import pytest

from ginjer_exercice.pipeline import step2_products
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.products import Color, DetectedProduct
from ginjer_exercice.schemas.step_outputs import (
    DetectedProductList,
    DetectedProductLLM,
    UniverseDetection,
    UniverseResult,
)

from .pipeline_fakes import FakeLLMProvider, FakePromptRegistry, FakeTraceSpan


@pytest.fixture
def chanel_ad() -> Ad:
    return Ad(
        platform_ad_id="test-ad-001",
        brand=Brand.CHANEL,
        texts=[AdText(title="CHANEL N°5", body_text="The legend.")],
        media_urls=["gs://bucket/n5.jpg"],
    )


@pytest.fixture
def universe_result() -> UniverseResult:
    return UniverseResult(
        detected_universes=[
            UniverseDetection(universe="Fragrance", confidence=0.95, reasoning="Perfume bottle visible.")
        ]
    )


@pytest.fixture
def single_product_list() -> DetectedProductList:
    return DetectedProductList(
        products=[
            DetectedProductLLM(
                raw_description="Transparent glass bottle labelled N°5",
                universe="Fragrance",
                color="Gold",
                importance=5,
            )
        ],
        overall_confidence=0.95,
    )


@pytest.fixture
def multi_product_list() -> DetectedProductList:
    return DetectedProductList(
        products=[
            DetectedProductLLM(
                raw_description="Main perfume bottle",
                universe="Fragrance",
                color="Gold",
                importance=5,
            ),
            DetectedProductLLM(
                raw_description="Background accessory",
                universe="Women",
                color="Black",
                importance=2,
            ),
        ],
        overall_confidence=0.85,
    )


@pytest.fixture
def zero_importance_list() -> DetectedProductList:
    """Liste avec un produit importance=0 — doit être filtré."""
    return DetectedProductList(
        products=[
            DetectedProductLLM(
                raw_description="Barely visible item in background",
                universe="Women",
                color="Beige",
                importance=0,
            )
        ],
        overall_confidence=0.3,
    )


class TestStep2Execute:

    def test_returns_list_of_detected_products(
        self, chanel_ad, universe_result, single_product_list
    ) -> None:
        result = step2_products.execute(
            chanel_ad, universe_result,
            llm_provider=FakeLLMProvider([single_product_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], DetectedProduct)

    def test_product_fields_correctly_mapped(
        self, chanel_ad, universe_result, single_product_list
    ) -> None:
        result = step2_products.execute(
            chanel_ad, universe_result,
            llm_provider=FakeLLMProvider([single_product_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        p = result[0]
        assert p.raw_description == "Transparent glass bottle labelled N°5"
        assert p.universe == "Fragrance"
        assert p.color == Color.GOLD
        assert p.importance == 5

    def test_importance_zero_filtered(
        self, chanel_ad, universe_result, zero_importance_list
    ) -> None:
        """Les produits avec importance=0 sont filtrés avant step3."""
        result = step2_products.execute(
            chanel_ad, universe_result,
            llm_provider=FakeLLMProvider([zero_importance_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result == []

    def test_results_sorted_by_importance_descending(
        self, chanel_ad, universe_result, multi_product_list
    ) -> None:
        result = step2_products.execute(
            chanel_ad, universe_result,
            llm_provider=FakeLLMProvider([multi_product_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        importances = [p.importance for p in result]
        assert importances == sorted(importances, reverse=True)

    def test_empty_product_list_returns_empty(
        self, chanel_ad, universe_result
    ) -> None:
        empty_list = DetectedProductList(products=[], overall_confidence=0.9)
        result = step2_products.execute(
            chanel_ad, universe_result,
            llm_provider=FakeLLMProvider([empty_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result == []

    def test_unknown_color_falls_back_to_multicolor(
        self, chanel_ad, universe_result
    ) -> None:
        weird_color_list = DetectedProductList(
            products=[
                DetectedProductLLM(
                    raw_description="A product",
                    universe="Women",
                    color="Chartreuse",  # Non supporté
                    importance=3,
                )
            ],
            overall_confidence=0.8,
        )
        result = step2_products.execute(
            chanel_ad, universe_result,
            llm_provider=FakeLLMProvider([weird_color_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result[0].color == Color.MULTICOLOR

    def test_prompt_fetched_by_name(
        self, chanel_ad, universe_result, single_product_list
    ) -> None:
        fake_registry = FakePromptRegistry()
        step2_products.execute(
            chanel_ad, universe_result,
            llm_provider=FakeLLMProvider([single_product_list]),
            prompt_registry=fake_registry,
            trace=FakeTraceSpan(),
        )
        assert "pipeline/products" in fake_registry.gets
