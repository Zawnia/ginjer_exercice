"""Exceptions métier et infrastructure de l'application.

Chaque module du data_access/ layer lève des exceptions spécifiques
pour que le domaine puisse les attraper explicitement.
"""


class GinjerException(Exception):
    """Exception de base pour l'application."""
    pass


# ──────────────────────────────────────────────────────────────
# Schemas / Domain
# ──────────────────────────────────────────────────────────────

class UnsupportedBrandError(GinjerException):
    """Levée quand une marque n'est pas supportée par le système."""
    pass


class TaxonomyNotFoundError(GinjerException):
    """Levée quand la taxonomie n'a pas pu être chargée ou trouvée."""
    pass


class LLMValidationError(GinjerException):
    """Levée quand la validation de la réponse LLM échoue après retries."""
    pass


# ──────────────────────────────────────────────────────────────
# BigQuery
# ──────────────────────────────────────────────────────────────

class AdNotFoundError(GinjerException):
    """Levée quand fetch_ad() ne trouve aucune ligne pour l'ad_id donné."""
    pass


class BigQueryAccessError(GinjerException):
    """Levée quand une requête BigQuery échoue pour des raisons d'infrastructure
    (auth, réseau, permissions, syntaxe SQL).
    """
    pass


# ──────────────────────────────────────────────────────────────
# Media Fetcher
# ──────────────────────────────────────────────────────────────

class MediaFetchError(GinjerException):
    """Erreur de base pour les échecs de téléchargement de médias."""
    pass


class MediaNotFoundError(MediaFetchError):
    """HTTP 404 ou ressource introuvable."""
    pass


class MediaTooLargeError(MediaFetchError):
    """Le contenu dépasse la limite de taille configurée."""
    pass


class MediaUnsupportedError(MediaFetchError):
    """Type MIME non supporté par le pipeline."""
    pass


# ──────────────────────────────────────────────────────────────
# Results Repository
# ──────────────────────────────────────────────────────────────

class RepositoryError(GinjerException):
    """Erreur de base pour les opérations de persistance."""
    pass
