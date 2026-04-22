from contextlib import contextmanager
from typing import Any, Generator
from pydantic import BaseModel

from .client import get_langfuse_client
from ..schemas.ad import Ad

def _safe_model_dump(obj: Any) -> Any:
    """Helper pour sérialiser les objets Pydantic de manière sécurisée."""
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _safe_model_dump(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_model_dump(item) for item in obj]
    return obj

class NullObservationContext:
    """Objet nul qui implémente l'interface d'observation Langfuse en no-op."""
    def update(self, **kwargs): pass
    def score(self, **kwargs): pass
    def end(self, **kwargs): pass

class NullSpanContext(NullObservationContext):
    """Objet nul pour les spans."""
    def update_trace(self, **kwargs): pass
    def score_trace(self, **kwargs): pass

class NullTraceContext(NullObservationContext):
    """Objet nul pour les traces."""
    def update_trace(self, **kwargs): pass

@contextmanager
def pipeline_trace(ad: Ad, session_id: str | None = None) -> Generator[Any, None, None]:
    """Ouvre la trace racine pour le traitement d'une publicité.
    
    Hiérarchie Langfuse :
        session = un batch CLI
        trace   = une pub (platform_ad_id)
        span    = une étape pipeline
    """
    langfuse = get_langfuse_client()
    if langfuse is None:
        yield NullTraceContext()
        return
    
    # Langfuse v4 OpenTelemetry
    with langfuse.start_as_current_observation(
        as_type="span",
        name=f"pipeline_{ad.platform_ad_id}",
        input=_safe_model_dump(ad),
    ) as root_span:
        # Propager les attributs de trace
        # L'API v4 utilise update() pour le span ou update_trace() (mais start_as_current_observation return un builder)
        # Note: on utilise des propriétés de trace sur l'observation.
        
        # Le root_span a les attributs de trace par défaut dans le SDK v4
        tags = [ad.brand.value]
        metadata = {
            "platform_ad_id": ad.platform_ad_id,
            "media_count": len(ad.media_urls),
            "text_count": len(ad.texts),
        }
        
        root_span.update_trace(
            session_id=session_id,
            user_id=ad.brand.value,
            tags=tags,
            metadata=metadata,
        )
        yield root_span


@contextmanager
def step_span(name: str, input_payload: Any = None) -> Generator[Any, None, None]:
    """Ouvre un span enfant pour une étape du pipeline."""
    langfuse = get_langfuse_client()
    if langfuse is None:
        yield NullSpanContext()
        return
    
    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input=_safe_model_dump(input_payload),
    ) as span:
        yield span
