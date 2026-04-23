from __future__ import annotations

from ginjer_exercice.pipeline.orchestrator import run_ad
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.pipeline import PipelineOutput
from ginjer_exercice.schemas.products import Color, DetectedProduct, ProductClassification, ProductName
from ginjer_exercice.schemas.step_outputs import UniverseDetection, UniverseResult


class FakeResultsRepository:
    def __init__(self) -> None:
        self.saved: list[PipelineOutput] = []

    def save(self, output: PipelineOutput) -> None:
        self.saved.append(output)


class FakeTaxonomy:
    def is_terminal_category(self, universe: str, category: str) -> bool:
        return False

    def is_valid_path(self, universe: str, category: str, subcategory: str) -> bool:
        return (universe, category, subcategory) == ("Fragrance", "Perfume", "Eau de parfum")


def test_run_ad_persists_pipeline_output(monkeypatch) -> None:
    ad = Ad(
        platform_ad_id="ad-123",
        brand=Brand.CHANEL,
        texts=[AdText(title="Chance", body_text="Fragrance campaign")],
        media_urls=["https://example.com/chanel.jpg"],
    )
    repository = FakeResultsRepository()

    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.load_taxonomy",
        lambda brand: FakeTaxonomy(),
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step1_universe.execute",
        lambda *args, **kwargs: UniverseResult(
            detected_universes=[UniverseDetection(universe="Fragrance", confidence=0.9, reasoning="bottle")]
        ),
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step2_products.execute",
        lambda *args, **kwargs: [
            DetectedProduct(
                raw_description="Square bottle",
                universe="Fragrance",
                color=Color.GOLD,
                importance=5,
            )
        ],
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step3_classify.execute",
        lambda *args, **kwargs: ProductClassification(
            universe="Fragrance",
            category="Perfume",
            subcategory="Eau de parfum",
            confidence=0.8,
        ),
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step4_name.execute",
        lambda *args, **kwargs: ProductName(
            name="Chance",
            source="explicit",
            confidence=0.6,
            needs_review=False,
            sources_consulted=["ad_title"],
        ),
    )

    output = run_ad(
        ad,
        llm_provider=object(),
        prompt_registry=object(),
        results_repository=repository,
    )

    assert len(repository.saved) == 1
    assert output == repository.saved[0]
    assert output.ad_id == "ad-123"
    assert output.scores.taxonomy_coherence == 1.0
    assert output.scores.confidence == 0.7
    assert output.needs_review is False


def test_run_ad_keeps_name_info_none_when_step4_finds_nothing(monkeypatch) -> None:
    ad = Ad(
        platform_ad_id="ad-456",
        brand=Brand.CHANEL,
        texts=[AdText(title="No explicit name")],
        media_urls=[],
    )
    repository = FakeResultsRepository()

    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.load_taxonomy",
        lambda brand: FakeTaxonomy(),
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step1_universe.execute",
        lambda *args, **kwargs: UniverseResult(
            detected_universes=[UniverseDetection(universe="Fragrance", confidence=0.9, reasoning="bottle")]
        ),
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step2_products.execute",
        lambda *args, **kwargs: [
            DetectedProduct(
                raw_description="Square bottle",
                universe="Fragrance",
                color=Color.GOLD,
                importance=5,
            )
        ],
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step3_classify.execute",
        lambda *args, **kwargs: ProductClassification(
            universe="Fragrance",
            category="Perfume",
            subcategory="Eau de parfum",
            confidence=0.8,
        ),
    )
    monkeypatch.setattr(
        "ginjer_exercice.pipeline.orchestrator.step4_name.execute",
        lambda *args, **kwargs: None,
    )

    output = run_ad(
        ad,
        llm_provider=object(),
        prompt_registry=object(),
        results_repository=repository,
    )

    assert output.products[0].name_info is None
    assert output.needs_review is True
    assert output.scores.confidence == 0.8
