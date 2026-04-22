"""Tests unitaires pour ResultsRepository (SQLite).

Utilise une base SQLite en mémoire pour des tests rapides et isolés.
"""

import sqlite3

import pytest

from src.ginjer_exercice.data_access.results_repository import SQLiteResultsRepository
from src.ginjer_exercice.schemas.ad import Brand
from src.ginjer_exercice.schemas.pipeline import PipelineOutput
from src.ginjer_exercice.schemas.products import (
    Color,
    DetectedProduct,
    FinalProductLabel,
    ProductClassification,
    ProductName,
)
from src.ginjer_exercice.schemas.scores import ScoreReport


# ── Helpers ────────────────────────────────────────────────────


def _make_output(
    ad_id: str = "ad_001",
    brand: Brand = Brand.CHANEL,
    needs_review: bool = False,
    trace_id: str = "trace_001",
) -> PipelineOutput:
    """Crée un PipelineOutput minimal pour les tests."""
    products = []
    if needs_review:
        products.append(
            FinalProductLabel(
                detected=DetectedProduct(
                    importance=3,
                    color=Color.BLACK,
                    universe="Fashion",
                    raw_description="Un sac noir",
                ),
                classification=ProductClassification(
                    universe="Fashion",
                    category="Bags",
                    subcategory="Handbags",
                    product_type="Shoulder Bag",
                    confidence=0.6,
                ),
                name_info=ProductName(
                    name=None,
                    source="fallback_llm",
                    confidence=0.4,
                    needs_review=True,
                    sources_consulted=["llm"],
                ),
            )
        )
    else:
        products.append(
            FinalProductLabel(
                detected=DetectedProduct(
                    importance=5,
                    color=Color.BLACK,
                    universe="Fashion",
                    raw_description="Classic Flap",
                ),
                classification=ProductClassification(
                    universe="Fashion",
                    category="Bags",
                    subcategory="Handbags",
                    product_type="Flap Bag",
                    confidence=0.95,
                ),
                name_info=ProductName(
                    name="Classic Flap",
                    source="explicit",
                    confidence=0.98,
                    needs_review=False,
                ),
            )
        )

    return PipelineOutput(
        ad_id=ad_id,
        brand=brand,
        products=products,
        scores=ScoreReport(taxonomy_coherence=0.9, confidence=0.85, llm_judge=0.8),
        trace_id=trace_id,
    )


@pytest.fixture
def repo():
    """SQLiteResultsRepository avec base en mémoire."""
    conn = sqlite3.connect(":memory:")
    return SQLiteResultsRepository(connection=conn)


# ── save() + get() ────────────────────────────────────────────


class TestSaveAndGet:
    """Tests pour save() et get()."""

    def test_save_then_get(self, repo):
        """Un résultat sauvegardé est récupérable via get()."""
        output = _make_output(ad_id="ad_42")
        repo.save(output)

        result = repo.get("ad_42")
        assert result is not None
        assert result.ad_id == "ad_42"
        assert result.brand == Brand.CHANEL
        assert len(result.products) == 1

    def test_get_nonexistent_returns_none(self, repo):
        """get() retourne None si absent."""
        result = repo.get("nonexistent")
        assert result is None

    def test_idempotent_save(self, repo):
        """Sauvegarder deux fois le même ad_id ne duplique pas."""
        output = _make_output(ad_id="ad_dup")
        repo.save(output)
        repo.save(output)

        result = repo.get("ad_dup")
        assert result is not None
        assert result.ad_id == "ad_dup"

    def test_save_overwrites_with_new_data(self, repo):
        """Un second save met à jour les données."""
        output1 = _make_output(ad_id="ad_upd", trace_id="trace_v1")
        repo.save(output1)

        output2 = _make_output(ad_id="ad_upd", trace_id="trace_v2")
        repo.save(output2)

        result = repo.get("ad_upd")
        assert result.trace_id == "trace_v2"


# ── exists() ──────────────────────────────────────────────────


class TestExists:
    """Tests pour exists()."""

    def test_exists_true(self, repo):
        """exists() retourne True si le résultat est persisté."""
        repo.save(_make_output(ad_id="ad_exists"))
        assert repo.exists("ad_exists") is True

    def test_exists_false(self, repo):
        """exists() retourne False si absent."""
        assert repo.exists("nope") is False


# ── list_needs_review() ──────────────────────────────────────


class TestListNeedsReview:
    """Tests pour list_needs_review()."""

    def test_empty_repo(self, repo):
        """Un repository vide retourne une liste vide."""
        assert repo.list_needs_review() == []

    def test_returns_only_review_flagged(self, repo):
        """Seuls les résultats avec needs_review=True sont retournés."""
        repo.save(_make_output(ad_id="ok_1", needs_review=False))
        repo.save(_make_output(ad_id="review_1", needs_review=True))
        repo.save(_make_output(ad_id="ok_2", needs_review=False))
        repo.save(_make_output(ad_id="review_2", needs_review=True))

        results = repo.list_needs_review()
        assert len(results) == 2
        review_ids = {r.ad_id for r in results}
        assert review_ids == {"review_1", "review_2"}

    def test_filter_by_brand(self, repo):
        """Le filtre par marque fonctionne correctement."""
        repo.save(_make_output(ad_id="chanel_review", brand=Brand.CHANEL, needs_review=True))
        repo.save(_make_output(ad_id="dior_review", brand=Brand.DIOR, needs_review=True))
        repo.save(_make_output(ad_id="chanel_ok", brand=Brand.CHANEL, needs_review=False))

        results = repo.list_needs_review(brand=Brand.CHANEL)
        assert len(results) == 1
        assert results[0].ad_id == "chanel_review"

    def test_filter_by_brand_no_results(self, repo):
        """Filtre par marque sans résultats retourne []."""
        repo.save(_make_output(ad_id="chanel_r", brand=Brand.CHANEL, needs_review=True))

        results = repo.list_needs_review(brand=Brand.DIOR)
        assert results == []


# ── PipelineOutput.needs_review property ──────────────────────


class TestNeedsReviewProperty:
    """Vérifie que la propriété calculée needs_review fonctionne."""

    def test_no_products_no_review(self):
        """Pas de produits → pas de review."""
        output = PipelineOutput(
            ad_id="empty",
            brand=Brand.CHANEL,
            products=[],
            scores=ScoreReport(),
            trace_id="t1",
        )
        assert output.needs_review is False

    def test_all_products_ok(self):
        output = _make_output(needs_review=False)
        assert output.needs_review is False

    def test_product_needs_review(self):
        output = _make_output(needs_review=True)
        assert output.needs_review is True

    def test_product_without_name_info(self):
        """Un produit sans name_info implique needs_review=True."""
        output = PipelineOutput(
            ad_id="no_name",
            brand=Brand.DIOR,
            products=[
                FinalProductLabel(
                    detected=DetectedProduct(
                        importance=3,
                        color=Color.BLACK,
                        universe="Fashion",
                        raw_description="test",
                    ),
                    classification=ProductClassification(
                        universe="Fashion",
                        category="Bags",
                        subcategory="Handbags",
                        product_type="Tote",
                        confidence=0.7,
                    ),
                    name_info=None,
                ),
            ],
            scores=ScoreReport(),
            trace_id="t1",
        )
        assert output.needs_review is True
