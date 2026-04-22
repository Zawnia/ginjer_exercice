import logging
from langfuse import Langfuse
from ..config import get_settings

logger = logging.getLogger(__name__)

def get_langfuse_client() -> Langfuse | None:
    """Point d'entrée unique pour obtenir le client Langfuse.
    
    Utilise le singleton intégré du SDK via get_client().
    Retourne None si Langfuse est désactivé ou mal configuré,
    ce qui permet au reste du pipeline de fonctionner sans observabilité.
    """
    settings = get_settings()

    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse est activé mais les clés (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY) sont manquantes. "
            "L'observabilité est désactivée."
        )
        return None

    try:
        # Initialiser Langfuse via le constructeur pour passer explicitement les paramètres,
        # puis le SDK le mettra en cache pour get_client()
        langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        return langfuse
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client Langfuse: {e}")
        return None
