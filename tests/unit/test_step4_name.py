"""Tests unitaires pour Step 4 — Explicit name extraction."""

from __future__ import annotations

import pytest

from ginjer_exercice.pipeline import step4_name
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.products import Color, DetectedProduct, ProductClassification, ProductName
from ginjer_exercice.schemas.step_outputs import ExtractedName

from .pipeline_fakes import FakeLLMProvider, FakePromptRegistry, FakeTraceSpan


@pytest.fixture
def chanel_ad() -> Ad:
    return Ad(
        platform_ad_id="test-ad-001",
        brand=Brand.CHANEL,
        texts=[AdText(title="CHANCE EAU TENDRE", body_text="The new fragrance.")],
        media_urls=["gs://bucket/chance.jpg"],
    )


@pytest.fixture
def perfume_product() -> DetectedProduct:
    return DetectedProduct(
        raw_description="Pink glass bottle labelled CHANCE EAU TENDRE",
        universe="Fragrance",
        color=Color.PINK,
        importance=5,
    )


@pytest.fixture
def classification() -> ProductClassification:
    return ProductClassification(
        universe="Fragrance",
        category="Fragrance",
        subcategory="Eau de Toilette",
        product_type=None,
        confidence=0.92,
    )


@pytest.fixture
def explicit_name_response() -> ExtractedName:
    return ExtractedName(name="Chance Eau Tendre", found_in="title", confidence=0.97)


@pytest.fixture
def no_name_response() -> ExtractedName:
    return ExtractedName(name=None, found_in="none", confidence=0.0)


@pytest.fixture
def image_text_response() -> ExtractedName:
    return ExtractedName(name="N°5", found_in="image_text", confidence=0.85)


class TestStep4Execute:

    def test_returns_product_name_when_found(
        self, perfume_product, classification, chanel_ad, explicit_name_response
    ) -> None:
        result = step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=FakeLLMProvider([explicit_name_response]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert isinstance(result, ProductName)
        assert result.name == "Chance Eau Tendre"

    def test_returns_none_when_no_name_found(
        self, perfume_product, classification, chanel_ad, no_name_response
    ) -> None:
        """Retourner None signale à l'orchestrator qu'il faut déclencher le fallback."""
        result = step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=FakeLLMProvider([no_name_response]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result is None

    def test_explicit_source_when_name_found(
        self, perfume_product, classification, chanel_ad, explicit_name_response
    ) -> None:
        result = step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=FakeLLMProvider([explicit_name_response]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result.source == "explicit"
        assert result.needs_review is False

    def test_sources_consulted_contains_found_in(
        self, perfume_product, classification, chanel_ad, explicit_name_response
    ) -> None:
        result = step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=FakeLLMProvider([explicit_name_response]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert "ad_title" in result.sources_consulted

    def test_image_text_source_accepted(
        self, perfume_product, classification, chanel_ad, image_text_response
    ) -> None:
        """Un nom visible dans l'image est aussi valide."""
        result = step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=FakeLLMProvider([image_text_response]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result is not None
        assert result.name == "N°5"
        assert "ad_image_text" in result.sources_consulted

    def test_confidence_propagated(
        self, perfume_product, classification, chanel_ad, explicit_name_response
    ) -> None:
        result = step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=FakeLLMProvider([explicit_name_response]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result.confidence == 0.97

    def test_llm_called_once(
        self, perfume_product, classification, chanel_ad, explicit_name_response
    ) -> None:
        fake_llm = FakeLLMProvider([explicit_name_response])
        step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=fake_llm,
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert fake_llm.call_count == 1

    def test_prompt_fetched_by_name(
        self, perfume_product, classification, chanel_ad, explicit_name_response
    ) -> None:
        fake_registry = FakePromptRegistry()
        step4_name.execute(
            perfume_product, classification, chanel_ad,
            llm_provider=FakeLLMProvider([explicit_name_response]),
            prompt_registry=fake_registry,
            trace=FakeTraceSpan(),
        )
        assert "pipeline/name_extraction" in fake_registry.gets
