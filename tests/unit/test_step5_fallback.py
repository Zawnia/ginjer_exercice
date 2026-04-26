from __future__ import annotations

from ginjer_exercice.data_access.catalog_provider import CatalogProvider, NullCatalogProvider
from ginjer_exercice.observability.runtime_warnings import collect_runtime_warnings
from ginjer_exercice.observability.tracing import NullTraceContext
from ginjer_exercice.pipeline.step5_fallback import step5_fallback
from ginjer_exercice.schemas.ad import Brand
from ginjer_exercice.schemas.pipeline import PipelineOutput
from ginjer_exercice.schemas.products import Color, DetectedProduct, FinalProductLabel, ProductClassification
from ginjer_exercice.schemas.scores import ScoreReport
from ginjer_exercice.schemas.step_outputs import FallbackNameSuggestion
from ginjer_exercice.web_search.null_provider import NullWebSearchProvider

from .pipeline_fakes import FakeLLMProvider, FakePromptRegistry


class StaticCatalogProvider(CatalogProvider):
    """Return a predefined subset for Step 5 tests."""

    def __init__(self, subset: list[dict[str, object]]) -> None:
        self.subset = subset

    def get_subset(self, brand: Brand, universe: str, category: str, limit: int = 50) -> list[dict[str, object]]:
        return self.subset[:limit]


def _sample_product() -> DetectedProduct:
    return DetectedProduct(
        raw_description="Transparent rectangular bottle with gold cap",
        universe="Beauty",
        color=Color.GOLD,
        importance=5,
    )


def _sample_classification() -> ProductClassification:
    return ProductClassification(
        universe="Beauty",
        category="Perfume",
        subcategory="Women's Perfume",
        product_type="Eau de parfum",
        confidence=0.94,
    )


def test_step5_prompt_receives_catalog_subset() -> None:
    fake_llm = FakeLLMProvider([FallbackNameSuggestion(name="CHANEL N5", confidence=0.95, reasoning="catalog match")])
    registry = FakePromptRegistry(
        prompt_override=(
            "Brand={{brand}}\nUniverse={{universe}}\nCategory={{category}}\nSubcategory={{subcategory}}\n"
            "ProductType={{product_type}}\nColor={{color}}\nImportance={{importance}}\n"
            "Visual={{visual_description}}\nAd={{ad_context}}\nCatalog={{catalog_subset}}\n"
        )
    )
    catalog = StaticCatalogProvider(
        [{"brand": "CHANEL", "universe": "Beauty", "category": "Perfume", "subcategory": "Women's Perfume"}]
    )

    step5_fallback(
        product=_sample_product(),
        classification=_sample_classification(),
        brand=Brand.CHANEL,
        ad_context="The icon of fragrance.",
        llm_provider=fake_llm,
        prompt_registry=registry,
        trace_context=NullTraceContext(),
        catalog_provider=catalog,
        web_search_provider=NullWebSearchProvider(),
    )

    prompt_text = fake_llm.calls[0]["messages"][0].text
    assert '"subcategory": "Women\'s Perfume"' in prompt_text


def test_step5_with_null_catalog_provider_does_not_fail() -> None:
    fake_llm = FakeLLMProvider([FallbackNameSuggestion(name="CHANEL N5", confidence=0.95, reasoning="strong fit")])

    result = step5_fallback(
        product=_sample_product(),
        classification=_sample_classification(),
        brand=Brand.CHANEL,
        ad_context="The icon of fragrance.",
        llm_provider=fake_llm,
        prompt_registry=FakePromptRegistry(
            prompt_override="Catalog={{catalog_subset}}\nName={{visual_description}}\n"
        ),
        trace_context=NullTraceContext(),
        catalog_provider=NullCatalogProvider(),
        web_search_provider=NullWebSearchProvider(),
    )

    assert result.name == "CHANEL N5"
    assert "Catalog=[]" in fake_llm.calls[0]["messages"][0].text


def test_step5_high_confidence_suggestion_adds_fallback_warning() -> None:
    with collect_runtime_warnings() as warnings:
        result = step5_fallback(
            product=_sample_product(),
            classification=_sample_classification(),
            brand=Brand.CHANEL,
            ad_context="The icon of fragrance.",
            llm_provider=FakeLLMProvider(
                [FallbackNameSuggestion(name="CHANEL N5", confidence=0.95, reasoning="strong fit")]
            ),
            prompt_registry=FakePromptRegistry(
                prompt_override="Visual={{visual_description}}\nCatalog={{catalog_subset}}\n"
            ),
            trace_context=NullTraceContext(),
            catalog_provider=NullCatalogProvider(),
            web_search_provider=NullWebSearchProvider(),
        )

    assert result.name == "CHANEL N5"
    assert result.source == "fallback_enriched"
    assert result.needs_review is False
    assert warnings == ["step5: name resolved via fallback (confidence=0.95, source=fallback_enriched)"]


def test_step5_low_confidence_suggestion_adds_warning_and_needs_review() -> None:
    with collect_runtime_warnings() as warnings:
        result = step5_fallback(
            product=_sample_product(),
            classification=_sample_classification(),
            brand=Brand.CHANEL,
            ad_context="The icon of fragrance.",
            llm_provider=FakeLLMProvider(
                [FallbackNameSuggestion(name="CHANEL N5", confidence=0.5, reasoning="weak fit")]
            ),
            prompt_registry=FakePromptRegistry(
                prompt_override="Visual={{visual_description}}\nCatalog={{catalog_subset}}\n"
            ),
            trace_context=NullTraceContext(),
            catalog_provider=NullCatalogProvider(),
            web_search_provider=NullWebSearchProvider(),
        )

    assert result.name == "CHANEL N5"
    assert result.source == "fallback_enriched"
    assert result.needs_review is True
    assert warnings == [
        "step5: low confidence name, flagged for review",
        "step5: name resolved via fallback (confidence=0.50, source=fallback_enriched)",
    ]


def test_step5_null_suggestion_fails_and_marks_pipeline_for_review() -> None:
    product = _sample_product()
    classification = _sample_classification()

    with collect_runtime_warnings() as warnings:
        result = step5_fallback(
            product=product,
            classification=classification,
            brand=Brand.CHANEL,
            ad_context="The icon of fragrance.",
            llm_provider=FakeLLMProvider(
                [FallbackNameSuggestion(name=None, confidence=0.0, reasoning="no distinctive model evidence")]
            ),
            prompt_registry=FakePromptRegistry(
                prompt_override="Visual={{visual_description}}\nCatalog={{catalog_subset}}\n"
            ),
            trace_context=NullTraceContext(),
            catalog_provider=NullCatalogProvider(),
            web_search_provider=NullWebSearchProvider(),
        )

    output = PipelineOutput(
        ad_id="ad-null-step5",
        brand=Brand.CHANEL,
        products=[
            FinalProductLabel(
                detected=product,
                classification=classification,
                name_info=result,
            )
        ],
        warnings=list(warnings),
        scores=ScoreReport(),
        trace_id="trace-test",
    )

    assert result.name is None
    assert result.source == "fallback_failed"
    assert result.confidence == 0.0
    assert result.needs_review is True
    assert warnings == [
        "step5: could not identify product name",
        "step5: name resolved via fallback (confidence=0.00, source=fallback_failed)",
    ]
    assert output.needs_review is True
