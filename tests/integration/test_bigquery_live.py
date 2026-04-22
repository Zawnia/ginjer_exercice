"""Tests d'intégration BigQuery — nécessitent un accès GCP réel.

Ces tests sont marqués ``live`` et ne doivent pas être exécutés en CI
sans les credentials GCP configurés.

Utilisation :
    uv run pytest tests/integration/test_bigquery_live.py -m live -v
"""

import pytest
from google.cloud import bigquery

from src.ginjer_exercice.data_access.bigquery_client import BigQueryClient

# Marquer tous les tests de ce module comme "live"
pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def bq_client():
    """Crée un BigQueryClient avec un vrai client BigQuery (ADC)."""
    client = bigquery.Client(project="ginjer-440122")
    return BigQueryClient(bq_client=client)


class TestBigQueryLive:
    """Tests d'intégration minimaux contre la vraie table BQ."""

    def test_count_ads_returns_positive(self, bq_client):
        """La table contient au moins une publicité."""
        count = bq_client.count_ads()
        assert count > 0, "La table BQ devrait contenir des publicités"

    def test_fetch_ads_by_brand_returns_list(self, bq_client):
        """fetch_ads_by_brand retourne une liste (potentiellement vide mais sans crash)."""
        from src.ginjer_exercice.schemas.ad import Brand
        ads = bq_client.fetch_ads_by_brand(Brand.CHANEL, limit=5)
        assert isinstance(ads, list)
        # Si des résultats existent, vérifier la structure
        if ads:
            ad = ads[0]
            assert ad.platform_ad_id
            assert ad.brand == Brand.CHANEL
