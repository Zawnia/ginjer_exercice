"""Repository de persistance des résultats du pipeline.

Définit l'interface abstraite ``ResultsRepository`` et fournit une
implémentation concrète ``SQLiteResultsRepository`` pour le stockage
des ``PipelineOutput``.

SQLite est choisi comme backend par défaut car il offre :
- zéro service externe à gérer
- support des requêtes pour ``list_needs_review``
- ``save`` idempotent via ``INSERT OR REPLACE``
- portabilité Docker (volume monté)

Usage::

    import sqlite3
    repo = SQLiteResultsRepository(connection=sqlite3.connect("results.db"))
    repo.save(pipeline_output)
    result = repo.get("some_ad_id")
"""

import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from ..exceptions import RepositoryError
from ..schemas.ad import Brand
from ..schemas.pipeline import PipelineOutput

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_results (
    ad_id TEXT PRIMARY KEY,
    brand TEXT NOT NULL,
    trace_id TEXT,
    has_needs_review INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""

_CREATE_INDEX_BRAND = """
CREATE INDEX IF NOT EXISTS idx_brand ON pipeline_results(brand)
"""

_CREATE_INDEX_REVIEW = """
CREATE INDEX IF NOT EXISTS idx_needs_review ON pipeline_results(has_needs_review)
"""


class ResultsRepository(ABC):
    """Interface abstraite pour la persistance des résultats du pipeline.

    Garde le pipeline découplé des détails de stockage.
    """

    @abstractmethod
    def save(self, output: PipelineOutput) -> None:
        """Persiste un résultat de pipeline (idempotent).

        Args:
            output: Le résultat à sauvegarder.

        Raises:
            RepositoryError: En cas d'erreur de persistance.
        """
        ...

    @abstractmethod
    def get(self, ad_id: str) -> PipelineOutput | None:
        """Récupère un résultat par son ``ad_id``.

        Args:
            ad_id: Identifiant de la publicité.

        Returns:
            Le résultat ou ``None`` si absent.
        """
        ...

    @abstractmethod
    def list_needs_review(self, brand: Brand | None = None) -> list[PipelineOutput]:
        """Liste les résultats nécessitant une relecture humaine.

        Args:
            brand: Filtre optionnel par marque.

        Returns:
            Liste de résultats avec ``needs_review == True``.
        """
        ...

    @abstractmethod
    def exists(self, ad_id: str) -> bool:
        """Vérifie rapidement si un résultat existe.

        Args:
            ad_id: Identifiant de la publicité.

        Returns:
            ``True`` si un résultat est persisté pour cet ID.
        """
        ...


class SQLiteResultsRepository(ResultsRepository):
    """Implémentation SQLite du repository de résultats.

    Le schéma utilise un ``has_needs_review`` pré-calculé au moment
    du ``save()`` pour rendre le filtrage en base efficace.

    Args:
        connection: Connexion ``sqlite3.Connection`` injectée.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Crée les tables et index si nécessaire."""
        try:
            cursor = self._conn.cursor()
            cursor.execute(_CREATE_TABLE_SQL)
            cursor.execute(_CREATE_INDEX_BRAND)
            cursor.execute(_CREATE_INDEX_REVIEW)
            self._conn.commit()
            logger.info("SQLite schema initialized")
        except sqlite3.Error as exc:
            raise RepositoryError(f"Échec de l'initialisation du schéma SQLite : {exc}") from exc

    def save(self, output: PipelineOutput) -> None:
        """Persiste un résultat de pipeline (idempotent via INSERT OR REPLACE).

        Le champ ``has_needs_review`` est calculé une seule fois au moment
        de l'écriture pour que le filtrage en lecture soit efficace.
        """
        start = time.monotonic()
        try:
            payload = output.model_dump_json()
            now = datetime.now(timezone.utc).isoformat()
            has_review = 1 if output.needs_review else 0

            self._conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_results
                    (ad_id, brand, trace_id, has_needs_review, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    output.ad_id,
                    output.brand.value,
                    output.trace_id,
                    has_review,
                    now,
                    payload,
                ),
            )
            self._conn.commit()

            latency_ms = (time.monotonic() - start) * 1000
            logger.info(
                "Pipeline result saved",
                extra={
                    "ad_id": output.ad_id,
                    "brand": output.brand.value,
                    "has_needs_review": bool(has_review),
                    "latency_ms": round(latency_ms, 1),
                },
            )
        except sqlite3.Error as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(
                "Failed to save pipeline result",
                extra={
                    "ad_id": output.ad_id,
                    "latency_ms": round(latency_ms, 1),
                    "error": str(exc),
                },
            )
            raise RepositoryError(f"Échec de la sauvegarde pour ad_id='{output.ad_id}' : {exc}") from exc

    def get(self, ad_id: str) -> PipelineOutput | None:
        """Récupère un résultat par son ad_id. Retourne None si absent."""
        start = time.monotonic()
        try:
            cursor = self._conn.execute(
                "SELECT payload FROM pipeline_results WHERE ad_id = ?",
                (ad_id,),
            )
            row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Échec de la lecture pour ad_id='{ad_id}' : {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000

        if row is None:
            logger.debug("Pipeline result not found", extra={"ad_id": ad_id, "latency_ms": round(latency_ms, 1)})
            return None

        logger.debug("Pipeline result retrieved", extra={"ad_id": ad_id, "latency_ms": round(latency_ms, 1)})
        return PipelineOutput.model_validate_json(row["payload"])

    def list_needs_review(self, brand: Brand | None = None) -> list[PipelineOutput]:
        """Liste les résultats nécessitant une review, avec filtre optionnel par marque."""
        start = time.monotonic()
        try:
            if brand is not None:
                cursor = self._conn.execute(
                    """
                    SELECT payload FROM pipeline_results
                    WHERE has_needs_review = 1 AND brand = ?
                    ORDER BY created_at DESC
                    """,
                    (brand.value,),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT payload FROM pipeline_results
                    WHERE has_needs_review = 1
                    ORDER BY created_at DESC
                    """,
                )
            rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Échec du listing needs_review : {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "list_needs_review completed",
            extra={
                "brand": brand.value if brand else "all",
                "count": len(rows),
                "latency_ms": round(latency_ms, 1),
            },
        )

        return [PipelineOutput.model_validate_json(row["payload"]) for row in rows]

    def exists(self, ad_id: str) -> bool:
        """Vérifie rapidement si un résultat existe pour cet ad_id."""
        try:
            cursor = self._conn.execute(
                "SELECT 1 FROM pipeline_results WHERE ad_id = ? LIMIT 1",
                (ad_id,),
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as exc:
            raise RepositoryError(f"Échec du check d'existence pour ad_id='{ad_id}' : {exc}") from exc
