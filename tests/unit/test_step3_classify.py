"""Tests unitaires pour Step 3 — Taxonomy classification."""

from __future__ import annotations

import pytest

from ginjer_exercice.exceptions import LLMValidationError
from ginjer_exercice.pipeline import step3_classify
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.products import Color, DetectedProduct, ProductClassification
from ginjer_exercice.schemas.taxonomy import BrandTaxonomy

from .pipeline_fakes import FakeLLMProvider, FakePromptRegistry, FakeTraceSpan


@pytest.fixture
def sample_taxonomy() -> BrandTaxonomy:
    return BrandTaxonomy(
        tree={
            "Women": {"Bags": ["Handbags", "Clutch", "Crossbody"], "Shoes": ["Heels"]},
            "Fragrance": {"Fragrance": ["Eau de Parfum", "Eau de Toilette"]},
        }
    )


@pytest.fixture
def chanel_ad() -> Ad:
    return Ad(
        platform_ad_id="test-ad-001",
        brand=Brand.CHANEL,
        texts=[AdText(title="Classic Flap Bag")],
        media_urls=["gs://bucket/bag.jpg"],
    )


@pytest.fixture
def handbag_product() -> DetectedProduct:
    return DetectedProduct(
        raw_description="Quilted black leather bag with gold chain strap",
        universe="Women",
        color=Color.BLACK,
        importance=5,
    )


@pytest.fixture
def valid_classification() -> ProductClassification:
    return ProductClassification(
        universe="Women", category="Bags", subcategory="Handbags",
        product_type=None, confidence=0.92,
    )


@pytest.fixture
def invalid_classification() -> ProductClassification:
    return ProductClassification(
        universe="Women", category="Bags", subcategory="Tote Bags",  # n'existe pas
        product_type=None, confidence=0.75,
    )


class TestStep3HappyPath:

    def test_returns_valid_classification(
        self, handbag_product, chanel_ad, sample_taxonomy, valid_classification
    ) -> None:
        result = step3_classify.execute(
            handbag_product, chanel_ad, sample_taxonomy,
            llm_provider=FakeLLMProvider([valid_classification]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert isinstance(result, ProductClassification)
        assert result.subcategory == "Handbags"

    def test_llm_called_once_on_valid(
        self, handbag_product, chanel_ad, sample_taxonomy, valid_classification
    ) -> None:
        fake_llm = FakeLLMProvider([valid_classification])
        step3_classify.execute(
            handbag_product, chanel_ad, sample_taxonomy,
            llm_provider=fake_llm,
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert fake_llm.call_count == 1

    def test_case_insensitive_match(
        self, handbag_product, chanel_ad, sample_taxonomy
    ) -> None:
        lowercase_cls = ProductClassification(
            universe="Women", category="Bags", subcategory="handbags",
            product_type=None, confidence=0.88,
        )
        fake_llm = FakeLLMProvider([lowercase_cls])
        result = step3_classify.execute(
            handbag_product, chanel_ad, sample_taxonomy,
            llm_provider=fake_llm,
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert fake_llm.call_count == 1  # Pas de retry
        assert result is not None

    def test_prompt_fetched_by_name(
        self, handbag_product, chanel_ad, sample_taxonomy, valid_classification
    ) -> None:
        fake_registry = FakePromptRegistry()
        step3_classify.execute(
            handbag_product, chanel_ad, sample_taxonomy,
            llm_provider=FakeLLMProvider([valid_classification]),
            prompt_registry=fake_registry,
            trace=FakeTraceSpan(),
        )
        assert "pipeline/classification" in fake_registry.gets


class TestStep3RetryMechanism:

    def test_invalid_first_valid_second(
        self, handbag_product, chanel_ad, sample_taxonomy,
        invalid_classification, valid_classification
    ) -> None:
        """Premier appel invalide, deuxième valide → retourne la classification valide."""
        fake_llm = FakeLLMProvider([invalid_classification, valid_classification])
        result = step3_classify.execute(
            handbag_product, chanel_ad, sample_taxonomy,
            llm_provider=fake_llm,
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        assert result.subcategory == "Handbags"
        assert fake_llm.call_count == 2

    def test_retry_appends_corrective_message(
        self, handbag_product, chanel_ad, sample_taxonomy,
        invalid_classification, valid_classification
    ) -> None:
        fake_llm = FakeLLMProvider([invalid_classification, valid_classification])
        step3_classify.execute(
            handbag_product, chanel_ad, sample_taxonomy,
            llm_provider=fake_llm,
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
        )
        # 2 appels ont été effectués
        assert fake_llm.call_count == 2
        # Le dernier message du 2ème appel contient le mot "INVALID" (message correctif)
        last_messages = fake_llm.calls[1]["messages"]
        corrective_texts = [m.text for m in last_messages if "INVALID" in m.text]
        assert len(corrective_texts) >= 1, (
            "Aucun message correctif 'INVALID' trouvé dans le 2ème appel LLM"
        )

    def test_all_invalid_raises_validation_error(
        self, handbag_product, chanel_ad, sample_taxonomy, invalid_classification
    ) -> None:
        """Toutes les tentatives invalides → LLMValidationError."""
        max_attempts = step3_classify.MAX_VALIDATION_RETRIES + 1
        fake_llm = FakeLLMProvider([invalid_classification] * max_attempts)

        with pytest.raises(LLMValidationError):
            step3_classify.execute(
                handbag_product, chanel_ad, sample_taxonomy,
                llm_provider=fake_llm,
                prompt_registry=FakePromptRegistry(),
                trace=FakeTraceSpan(),
            )
        assert fake_llm.call_count == max_attempts
