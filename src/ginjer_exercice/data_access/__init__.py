"""data_access — Couche d'accès aux données (I/O uniquement).

Ce package contient les trois composants d'I/O du pipeline :
- ``BigQueryClient`` : lecture des publicités depuis BigQuery.
- ``MediaFetcher`` : téléchargement des médias (images / vidéos).
- ``ResultsRepository`` : persistance des résultats du pipeline.

Règles transversales :
- Tout retour est un modèle Pydantic validé.
- Aucune logique métier dans ce package.
- Exceptions explicites et typées.
- Dépendances injectées via ``__init__``.
- Logging structuré sur chaque opération d'I/O.
"""

from .bigquery_client import BigQueryClient
from .media_fetcher import MediaFetcher
from .results_repository import ResultsRepository, SQLiteResultsRepository

__all__ = [
    "BigQueryClient",
    "MediaFetcher",
    "ResultsRepository",
    "SQLiteResultsRepository",
]
