"""Tests unitaires pour BigQueryClient.

Ces tests mockent entièrement le client BigQuery
pour valider la logique de conversion, normalisation et gestion d'erreurs.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.ginjer_exercice.data_access.bigquery_client import BigQueryClient, _row_to_ad
from src.ginjer_exercice.schemas.ad import Ad, AdText, Brand
from src.ginjer_exercice.schemas.helpers import normalize_brand
from src.ginjer_exercice.exceptions import (
    AdNotFoundError,
    BigQueryAccessError,
    UnsupportedBrandError,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_row(
    platform_ad_id: str = "ad_001",
    brand: str = "chanel",
    texts: list[dict] | None = None,
    media_urls: list[str] | None = None,
) -> dict:
    """Crée un dictionnaire simulant une ligne BigQuery."""
    return {
        "platform_ad_id": platform_ad_id,
        "brand": brand,
        "texts": texts,
        "media_urls": media_urls,
    }


def _make_bq_client_mock(rows: list[dict]) -> MagicMock:
    """Crée un mock de bigquery.Client qui retourne les lignes données."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.result.return_value = rows
    mock_client.query.return_value = mock_result
    return mock_client


# ── _row_to_ad tests ──────────────────────────────────────────


class TestRowToAd:
    """Tests pour la conversion de lignes BigQuery en modèles Ad."""

    def test_full_row_converts(self):
        """Une ligne complète produit un Ad avec tous les champs."""
        row = _make_row(
            texts=[
                {"title": "Mon Titre", "body_text": "Un texte", "caption": "Légende", "url": "http://ex.com"}
            ],
            media_urls=["http://img.jpg"],
        )
        ad = _row_to_ad(row)
        assert ad.platform_ad_id == "ad_001"
        assert ad.brand == Brand.CHANEL
        assert len(ad.texts) == 1
        assert ad.texts[0].title == "Mon Titre"
        assert ad.media_urls == ["http://img.jpg"]

    def test_texts_none(self):
        """texts=None produit une liste vide."""
        row = _make_row(texts=None)
        ad = _row_to_ad(row)
        assert ad.texts == []

    def test_texts_empty(self):
        """texts=[] produit une liste vide."""
        row = _make_row(texts=[])
        ad = _row_to_ad(row)
        assert ad.texts == []

    def test_texts_with_none_entry(self):
        """Les entrées None dans texts sont ignorées."""
        row = _make_row(texts=[None, {"title": "Valide", "body_text": None, "caption": None, "url": None}])
        ad = _row_to_ad(row)
        assert len(ad.texts) == 1
        assert ad.texts[0].title == "Valide"

    def test_nested_none_fields(self):
        """Les champs None dans un struct de texte sont acceptés."""
        row = _make_row(texts=[{"title": None, "body_text": None, "caption": None, "url": None}])
        ad = _row_to_ad(row)
        assert len(ad.texts) == 1
        assert ad.texts[0].title is None
        assert ad.texts[0].body_text is None

    def test_media_urls_none(self):
        """media_urls=None produit une liste vide."""
        row = _make_row(media_urls=None)
        ad = _row_to_ad(row)
        assert ad.media_urls == []

    def test_unknown_brand_raises(self):
        """Une marque inconnue lève UnsupportedBrandError."""
        row = _make_row(brand="unknown_brand")
        with pytest.raises(UnsupportedBrandError):
            _row_to_ad(row)


# ── Brand normalization tests ─────────────────────────────────


class TestBrandNormalization:
    """Tests pour la normalisation des valeurs de marques brutes."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("chanel", Brand.CHANEL),
            ("CHANEL", Brand.CHANEL),
            ("Chanel", Brand.CHANEL),
            ("dior", Brand.DIOR),
            ("DIOR", Brand.DIOR),
            ("louis vuitton", Brand.LOUIS_VUITTON),
            ("Louis_Vuitton", Brand.LOUIS_VUITTON),
            ("LOUIS_VUITTON", Brand.LOUIS_VUITTON),
            ("lv", Brand.LOUIS_VUITTON),
            ("balenciaga", Brand.BALENCIAGA),
            ("BALENCIAGA", Brand.BALENCIAGA),
            ("mfk", Brand.MFK),
            ("MFK", Brand.MFK),
            ("maison francis kurkdjian", Brand.MFK),
        ],
    )
    def test_known_aliases(self, raw: str, expected: Brand):
        assert normalize_brand(raw) == expected

    def test_unknown_brand_raises(self):
        with pytest.raises(UnsupportedBrandError):
            normalize_brand("totally_unknown")

    def test_whitespace_stripped(self):
        assert normalize_brand("  chanel  ") == Brand.CHANEL


# ── BigQueryClient tests ──────────────────────────────────────


class TestFetchAd:
    """Tests pour BigQueryClient.fetch_ad()."""

    def test_existing_ad_returns_ad(self):
        """fetch_ad retourne un Ad pour un ID existant."""
        row = _make_row(
            platform_ad_id="ad_123",
            brand="dior",
            texts=[{"title": "Test", "body_text": None, "caption": None, "url": None}],
            media_urls=["http://img.jpg"],
        )
        mock_bq = _make_bq_client_mock([row])
        client = BigQueryClient(bq_client=mock_bq)

        ad = client.fetch_ad("ad_123")
        assert isinstance(ad, Ad)
        assert ad.platform_ad_id == "ad_123"
        assert ad.brand == Brand.DIOR
        mock_bq.query.assert_called_once()

    def test_missing_ad_raises(self):
        """fetch_ad lève AdNotFoundError pour un ID absent."""
        mock_bq = _make_bq_client_mock([])
        client = BigQueryClient(bq_client=mock_bq)

        with pytest.raises(AdNotFoundError):
            client.fetch_ad("nonexistent")

    def test_infrastructure_error_raises(self):
        """Une erreur BigQuery est wrappée en BigQueryAccessError."""
        mock_bq = MagicMock()
        mock_bq.query.side_effect = Exception("connection refused")
        client = BigQueryClient(bq_client=mock_bq)

        with pytest.raises(BigQueryAccessError):
            client.fetch_ad("ad_123")


class TestFetchAdsByBrand:
    """Tests pour BigQueryClient.fetch_ads_by_brand()."""

    def test_returns_list(self):
        """fetch_ads_by_brand retourne une liste d'Ad."""
        rows = [
            _make_row(platform_ad_id="ad_1", brand="chanel"),
            _make_row(platform_ad_id="ad_2", brand="chanel"),
        ]
        mock_bq = _make_bq_client_mock(rows)
        client = BigQueryClient(bq_client=mock_bq)

        ads = client.fetch_ads_by_brand(Brand.CHANEL)
        assert len(ads) == 2
        assert all(isinstance(a, Ad) for a in ads)

    def test_empty_result_returns_empty_list(self):
        """Aucun résultat retourne []."""
        mock_bq = _make_bq_client_mock([])
        client = BigQueryClient(bq_client=mock_bq)

        ads = client.fetch_ads_by_brand(Brand.DIOR)
        assert ads == []


class TestFetchBatch:
    """Tests pour BigQueryClient.fetch_batch()."""

    def test_returns_found_ads(self):
        """fetch_batch retourne uniquement les ads trouvées."""
        rows = [_make_row(platform_ad_id="ad_1", brand="chanel")]
        mock_bq = _make_bq_client_mock(rows)
        client = BigQueryClient(bq_client=mock_bq)

        ads = client.fetch_batch(["ad_1", "ad_missing"])
        assert len(ads) == 1
        assert ads[0].platform_ad_id == "ad_1"

    def test_empty_ids_returns_empty(self):
        """fetch_batch([]) retourne []."""
        mock_bq = MagicMock()
        client = BigQueryClient(bq_client=mock_bq)

        ads = client.fetch_batch([])
        assert ads == []
        mock_bq.query.assert_not_called()


class TestCountAds:
    """Tests pour BigQueryClient.count_ads()."""

    def test_count_all(self):
        """count_ads sans filtre retourne le count."""
        mock_bq = _make_bq_client_mock([{"cnt": 42}])
        client = BigQueryClient(bq_client=mock_bq)

        assert client.count_ads() == 42

    def test_count_by_brand(self):
        """count_ads avec filtre marque."""
        mock_bq = _make_bq_client_mock([{"cnt": 10}])
        client = BigQueryClient(bq_client=mock_bq)

        assert client.count_ads(brand=Brand.CHANEL) == 10
