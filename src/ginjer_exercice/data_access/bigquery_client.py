"""Client BigQuery typé pour la lecture des publicités.

Ce module fournit un accès en lecture seule à la table BigQuery
``ginjer-440122.ia_eng_interview.ads``. Toutes les lignes sont
converties en modèles ``Ad`` validés avant de traverser la frontière
vers le domaine.

Authentification : Google ADC (``gcloud auth application-default login``).

Usage::

    from google.cloud import bigquery
    client = BigQueryClient(bq_client=bigquery.Client(project="ginjer-440122"))
    ad = client.fetch_ad("some_platform_ad_id")
"""

import logging
import time
from typing import Any

from google.cloud import bigquery
from google.cloud.bigquery import ArrayQueryParameter, ScalarQueryParameter

from ..exceptions import AdNotFoundError, BigQueryAccessError, UnsupportedBrandError
from ..schemas.ad import Ad, AdText, Brand
from ..schemas.helpers import normalize_brand

logger = logging.getLogger(__name__)

_TABLE = "`ginjer-440122.ia_eng_interview.ads`"


def _row_to_ad(row: Any) -> Ad:
    """Convertit une ligne BigQuery brute en modèle ``Ad`` validé.

    Args:
        row: Ligne retournée par ``bigquery.Client.query()``.

    Returns:
        Instance ``Ad`` construite et validée.

    Raises:
        UnsupportedBrandError: Si la marque brute n'est pas reconnue.
    """
    brand = normalize_brand(row["brand"])

    texts: list[AdText] = []
    raw_texts = row.get("texts") or []
    for t in raw_texts:
        if t is None:
            continue
        # BigQuery STRUCT fields — handle both dict and Row-like access
        if isinstance(t, dict):
            texts.append(AdText(
                body_text=t.get("body_text"),
                title=t.get("title"),
                caption=t.get("caption"),
                url=t.get("url"),
            ))
        else:
            # google.cloud.bigquery.table.Row — supports dict-like access
            texts.append(AdText(
                body_text=t.get("body_text"),
                title=t.get("title"),
                caption=t.get("caption"),
                url=t.get("url"),
            ))

    media_urls: list[str] = list(row.get("media_urls") or [])

    return Ad(
        platform_ad_id=row["platform_ad_id"],
        brand=brand,
        texts=texts,
        media_urls=media_urls,
    )


class BigQueryClient:
    """Client en lecture seule pour les publicités stockées dans BigQuery.

    Args:
        bq_client: Instance ``google.cloud.bigquery.Client`` injectée.
    """

    def __init__(self, bq_client: bigquery.Client) -> None:
        self._client = bq_client

    # ── Public API ──────────────────────────────────────────────

    def fetch_ad(self, ad_id: str) -> Ad:
        """Récupère une publicité par son ``platform_ad_id``.

        Args:
            ad_id: Identifiant unique de la publicité.

        Returns:
            Instance ``Ad`` validée.

        Raises:
            AdNotFoundError: Si aucune ligne ne correspond à cet ID.
            BigQueryAccessError: En cas d'erreur d'infrastructure.
        """
        query = f"""
            SELECT platform_ad_id, brand, texts, media_urls
            FROM {_TABLE}
            WHERE platform_ad_id = @ad_id
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                ScalarQueryParameter("ad_id", "STRING", ad_id),
            ]
        )

        start = time.monotonic()
        try:
            rows = list(self._client.query(query, job_config=job_config).result())
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(
                "BigQuery query failed",
                extra={"ad_id": ad_id, "latency_ms": round(latency_ms, 1), "error": str(exc)},
            )
            raise BigQueryAccessError(f"Échec de la requête BigQuery : {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "fetch_ad completed",
            extra={"ad_id": ad_id, "latency_ms": round(latency_ms, 1), "found": len(rows) > 0},
        )

        if not rows:
            raise AdNotFoundError(f"Aucune publicité trouvée pour ad_id='{ad_id}'")

        return _row_to_ad(rows[0])

    def fetch_ads_by_brand(self, brand: Brand, limit: int = 100) -> list[Ad]:
        """Récupère les publicités d'une marque donnée.

        Args:
            brand: La marque à filtrer.
            limit: Nombre maximal de résultats (défaut 100).

        Returns:
            Liste d'instances ``Ad``. Peut être vide si aucune pub trouvée.

        Raises:
            BigQueryAccessError: En cas d'erreur d'infrastructure.
        """
        query = f"""
            SELECT platform_ad_id, brand, texts, media_urls
            FROM {_TABLE}
            WHERE LOWER(brand) = @brand
            ORDER BY platform_ad_id
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                ScalarQueryParameter("brand", "STRING", brand.value.lower()),
                ScalarQueryParameter("limit", "INT64", limit),
            ]
        )

        start = time.monotonic()
        try:
            rows = list(self._client.query(query, job_config=job_config).result())
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(
                "BigQuery query failed",
                extra={"brand": brand.value, "latency_ms": round(latency_ms, 1), "error": str(exc)},
            )
            raise BigQueryAccessError(f"Échec de la requête BigQuery : {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "fetch_ads_by_brand completed",
            extra={
                "brand": brand.value,
                "count": len(rows),
                "limit": limit,
                "latency_ms": round(latency_ms, 1),
            },
        )

        ads: list[Ad] = []
        for row in rows:
            try:
                ads.append(_row_to_ad(row))
            except UnsupportedBrandError:
                logger.warning(
                    "Skipping row with unsupported brand",
                    extra={"platform_ad_id": row.get("platform_ad_id"), "brand": row.get("brand")},
                )
        return ads

    def fetch_batch(self, ad_ids: list[str]) -> list[Ad]:
        """Récupère un lot de publicités en une seule requête.

        Les IDs manquants sont ignorés avec un avertissement dans les logs.

        Args:
            ad_ids: Liste d'identifiants de publicités.

        Returns:
            Liste d'instances ``Ad`` trouvées.

        Raises:
            BigQueryAccessError: En cas d'erreur d'infrastructure.
        """
        if not ad_ids:
            return []

        query = f"""
            SELECT platform_ad_id, brand, texts, media_urls
            FROM {_TABLE}
            WHERE platform_ad_id IN UNNEST(@ad_ids)
            ORDER BY platform_ad_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                ArrayQueryParameter("ad_ids", "STRING", ad_ids),
            ]
        )

        start = time.monotonic()
        try:
            rows = list(self._client.query(query, job_config=job_config).result())
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(
                "BigQuery batch query failed",
                extra={"ad_ids_count": len(ad_ids), "latency_ms": round(latency_ms, 1), "error": str(exc)},
            )
            raise BigQueryAccessError(f"Échec de la requête BigQuery batch : {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000

        ads: list[Ad] = []
        for row in rows:
            try:
                ads.append(_row_to_ad(row))
            except UnsupportedBrandError:
                logger.warning(
                    "Skipping row with unsupported brand in batch",
                    extra={"platform_ad_id": row.get("platform_ad_id"), "brand": row.get("brand")},
                )

        found_ids = {a.platform_ad_id for a in ads}
        missing_ids = set(ad_ids) - found_ids
        if missing_ids:
            logger.warning(
                "Some ad_ids not found in BigQuery",
                extra={"missing_ids": sorted(missing_ids), "found_count": len(ads)},
            )

        logger.info(
            "fetch_batch completed",
            extra={
                "requested": len(ad_ids),
                "found": len(ads),
                "missing": len(missing_ids),
                "latency_ms": round(latency_ms, 1),
            },
        )
        return ads

    def count_ads(self, brand: Brand | None = None) -> int:
        """Compte le nombre de publicités, optionnellement filtrées par marque.

        Args:
            brand: Si fourni, filtre par cette marque.

        Returns:
            Nombre de publicités.

        Raises:
            BigQueryAccessError: En cas d'erreur d'infrastructure.
        """
        if brand is not None:
            query = f"""
                SELECT COUNT(*) AS cnt
                FROM {_TABLE}
                WHERE LOWER(brand) = @brand
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    ScalarQueryParameter("brand", "STRING", brand.value.lower()),
                ]
            )
        else:
            query = f"SELECT COUNT(*) AS cnt FROM {_TABLE}"
            job_config = bigquery.QueryJobConfig()

        start = time.monotonic()
        try:
            rows = list(self._client.query(query, job_config=job_config).result())
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(
                "BigQuery count query failed",
                extra={"brand": brand.value if brand else None, "latency_ms": round(latency_ms, 1), "error": str(exc)},
            )
            raise BigQueryAccessError(f"Échec du count BigQuery : {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        count = rows[0]["cnt"] if rows else 0
        logger.info(
            "count_ads completed",
            extra={"brand": brand.value if brand else "all", "count": count, "latency_ms": round(latency_ms, 1)},
        )
        return count
