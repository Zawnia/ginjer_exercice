"""Tests unitaires pour Step 1 — Universe detection.

Tous les appels LLM sont mockés via FakeLLMProvider.
Aucun appel réseau ne doit être effectué dans ces tests.
"""

from __future__ import annotations

import pytest

from ginjer_exercice.pipeline import step1_universe
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.step_outputs import UniverseDetection, UniverseResult

from .pipeline_fakes import FakeLLMProvider, FakePromptRegistry, FakeTraceSpan


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def chanel_ad() -> Ad:
    """Pub Chanel minimaliste avec texte et un média."""
    return Ad(
        platform_ad_id="test-ad-001",
        brand=Brand.CHANEL,
        texts=[
            AdText(
                title="CHANEL N°5",
                body_text="The legend continues.",
                caption="Discover the iconic fragrance",
                url="https://chanel.com/fragrance/n5",
            )
        ],
        media_urls=["gs://bucket/chanel_n5_video.mp4"],
    )


@pytest.fixture
def ad_no_media() -> Ad:
    """Pub sans média — uniquement du texte."""
    return Ad(
        platform_ad_id="test-ad-002",
        brand=Brand.DIOR,
        texts=[AdText(title="SAUVAGE", body_text="The new eau de parfum.")],
        media_urls=[],
    )


@pytest.fixture
def ad_empty() -> Ad:
    """Pub sans texte ni média — cas limite."""
    return Ad(
        platform_ad_id="test-ad-003",
        brand=Brand.BALENCIAGA,
        texts=[],
        media_urls=[],
    )


@pytest.fixture
def universe_result_fragrance() -> UniverseResult:
    """Réponse canned : pub fragrance avec haute confiance."""
    return UniverseResult(
        detected_universes=[
            UniverseDetection(
                universe="Fragrance",
                confidence=0.95,
                reasoning="Title explicitly mentions 'N°5', a known Chanel fragrance.",
            )
        ]
    )


@pytest.fixture
def universe_result_multi() -> UniverseResult:
    """Réponse canned : pub multi-univers (Fragrance + Women)."""
    return UniverseResult(
        detected_universes=[
            UniverseDetection(
                universe="Fragrance",
                confidence=0.9,
                reasoning="Clear perfume bottle in visual.",
            ),
            UniverseDetection(
                universe="Women",
                confidence=0.6,
                reasoning="Model wearing luxury dress in background.",
            ),
        ]
    )


# ──────────────────────────────────────────────────────────────
# Tests principaux
# ──────────────────────────────────────────────────────────────


class TestStep1Execute:
    """Tests de la fonction step1_universe.execute()."""

    def test_returns_universe_result(
        self,
        chanel_ad: Ad,
        universe_result_fragrance: UniverseResult,
    ) -> None:
        """L'output est bien un UniverseResult typé."""
        fake_llm = FakeLLMProvider([universe_result_fragrance])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        result = step1_universe.execute(
            chanel_ad,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        assert isinstance(result, UniverseResult)
        assert len(result.detected_universes) == 1
        assert result.detected_universes[0].universe == "Fragrance"

    def test_llm_called_exactly_once(
        self,
        chanel_ad: Ad,
        universe_result_fragrance: UniverseResult,
    ) -> None:
        """Le LLM n'est appelé qu'une seule fois par step."""
        fake_llm = FakeLLMProvider([universe_result_fragrance])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        step1_universe.execute(
            chanel_ad,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        assert fake_llm.call_count == 1

    def test_prompt_fetched_by_name(
        self,
        chanel_ad: Ad,
        universe_result_fragrance: UniverseResult,
    ) -> None:
        """Le bon prompt est demandé au registre."""
        fake_llm = FakeLLMProvider([universe_result_fragrance])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        step1_universe.execute(
            chanel_ad,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        assert "pipeline/universe" in fake_registry.gets

    def test_results_sorted_by_confidence_descending(
        self,
        chanel_ad: Ad,
        universe_result_multi: UniverseResult,
    ) -> None:
        """Les univers sont triés par confiance décroissante."""
        fake_llm = FakeLLMProvider([universe_result_multi])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        result = step1_universe.execute(
            chanel_ad,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        confidences = [u.confidence for u in result.detected_universes]
        assert confidences == sorted(confidences, reverse=True)

    def test_primary_universe_is_highest_confidence(
        self,
        chanel_ad: Ad,
        universe_result_multi: UniverseResult,
    ) -> None:
        """primary_universe retourne l'univers avec la plus haute confiance."""
        fake_llm = FakeLLMProvider([universe_result_multi])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        result = step1_universe.execute(
            chanel_ad,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        assert result.primary_universe == "Fragrance"

    def test_works_with_no_media(
        self,
        ad_no_media: Ad,
        universe_result_fragrance: UniverseResult,
    ) -> None:
        """Step 1 fonctionne avec une pub sans média (texte seul)."""
        fake_llm = FakeLLMProvider([universe_result_fragrance])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        result = step1_universe.execute(
            ad_no_media,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        assert isinstance(result, UniverseResult)
        # Vérifier que les messages LLM ont été construits sans lever d'exception
        assert fake_llm.call_count == 1

    def test_empty_ad_returns_empty_universes(
        self,
        ad_empty: Ad,
    ) -> None:
        """Une pub sans contenu peut retourner 0 univers — le LLM décide."""
        empty_result = UniverseResult(detected_universes=[])
        fake_llm = FakeLLMProvider([empty_result])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        result = step1_universe.execute(
            ad_empty,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        assert result.detected_universes == []
        assert result.primary_universe is None

    def test_universe_names_property(
        self,
        chanel_ad: Ad,
        universe_result_multi: UniverseResult,
    ) -> None:
        """universe_names retourne la liste des noms d'univers."""
        fake_llm = FakeLLMProvider([universe_result_multi])
        fake_registry = FakePromptRegistry()
        fake_trace = FakeTraceSpan()

        result = step1_universe.execute(
            chanel_ad,
            llm_provider=fake_llm,
            prompt_registry=fake_registry,
            trace=fake_trace,
        )

        assert "Fragrance" in result.universe_names
        assert "Women" in result.universe_names
