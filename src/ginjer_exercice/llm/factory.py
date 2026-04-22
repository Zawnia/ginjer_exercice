from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Retourne l'instance du provider LLM demandé.
    
    Args:
        provider_name: 'gemini' ou 'openai'
        **kwargs: Arguments supplémentaires (ex: api_key, project_id)
        
    Returns:
        Une instance de LLMProvider
        
    Raises:
        ValueError: Si le provider demandé n'est pas supporté.
    """
    provider_name = provider_name.lower()
    
    if provider_name == "gemini":
        return GeminiProvider(
            use_vertex=kwargs.get("use_vertex", False),
            project_id=kwargs.get("project_id"),
            location=kwargs.get("location", "us-central1")
        )
    elif provider_name == "openai":
        return OpenAIProvider(
            api_key=kwargs.get("api_key")
        )
    else:
        raise ValueError(f"Provider non supporté: {provider_name}")
