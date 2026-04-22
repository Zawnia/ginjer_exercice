from .client import get_langfuse_client
from ..taxonomy.store import TaxonomyStore
from ..schemas.ad import Brand
import logging

logger = logging.getLogger(__name__)

def score_taxonomy_coherence(
    brand: Brand,
    universe: str,
    category: str,
    subcategory: str,
    trace_id: str | None = None,
    observation_id: str | None = None,
) -> float:
    """Calcule et publie le score de cohérence taxonomique.
    
    Score déterministe : vérifie que le tuple (universe, category, subcategory)
    existe dans la taxonomie pour la marque donnée.
    Retourne 1.0 si valide, 0.0 sinon.
    Publie le score dans Langfuse si un trace_id est fourni.
    """
    store = TaxonomyStore()
    try:
        taxo = store.get_taxonomy(brand.value)
    except FileNotFoundError:
        # Fallback sur la canonique si la marque n'a pas de taxo spécifique compilée
        try:
            taxo = store.get_taxonomy("canonical")
        except FileNotFoundError:
            logger.error("Aucune taxonomie trouvée pour calculer le score de cohérence.")
            return 0.0
            
    is_valid = taxo.validate_path(universe, category, subcategory)
    score_value = 1.0 if is_valid else 0.0
    
    langfuse = get_langfuse_client()
    if langfuse is not None and trace_id is not None:
        try:
            langfuse.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name="taxonomy_coherence",
                value=score_value,
                comment=f"[{brand.value}] {universe} > {category} > {subcategory}"
            )
        except Exception as e:
            logger.warning(f"Impossible de logger le score dans Langfuse: {e}")
            
    return score_value

def score_confidence(
    step_name: str,
    confidence: float,
    trace_id: str | None = None,
    observation_id: str | None = None,
) -> float:
    """Publie un score de confiance par étape.
    
    Relaie directement la valeur de confidence renvoyée par le LLM.
    """
    langfuse = get_langfuse_client()
    if langfuse is not None and trace_id is not None:
        try:
            langfuse.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=f"confidence_{step_name}",
                value=confidence
            )
        except Exception as e:
            logger.warning(f"Impossible de logger le score de confiance dans Langfuse: {e}")
            
    return confidence

def score_llm_judge(
    trace_id: str | None = None,
    observation_id: str | None = None,
    **kwargs,
) -> float:
    """Score LLM-as-judge — stub pour phase 7.
    
    Raises:
        NotImplementedError: Toujours, en attendant l'implémentation phase 7.
    """
    raise NotImplementedError("LLM-as-judge sera implémenté en phase 7")
