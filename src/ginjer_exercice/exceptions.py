class GinjerException(Exception):
    """Exception de base pour l'application."""
    pass

class UnsupportedBrandError(GinjerException):
    """Levée quand une marque n'est pas supportée par le système."""
    pass

class TaxonomyNotFoundError(GinjerException):
    """Levée quand la taxonomie n'a pas pu être chargée ou trouvée."""
    pass

class LLMValidationError(GinjerException):
    """Levée quand la validation de la réponse LLM échoue après retries."""
    pass

class MediaFetchError(GinjerException):
    """Levée quand un média n'a pas pu être téléchargé."""
    pass
