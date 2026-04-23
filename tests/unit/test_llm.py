import pytest
from pydantic import BaseModel

from ginjer_exercice.llm.base import LLMCallConfig, LLMMessage, MediaPart, TextPart, LLMProvider
from ginjer_exercice.llm.factory import get_provider
from ginjer_exercice.llm.gemini_provider import GeminiProvider
from ginjer_exercice.llm.openai_provider import OpenAIProvider

class DummyResponseModel(BaseModel):
    name: str
    confidence: float

def test_get_provider_gemini(monkeypatch):
    # 1. On "fake" la variable d'environnement pour que le SDK Google soit content
    monkeypatch.setenv("GEMINI_API_KEY", "cle_api_factice_pour_le_test")
    
    provider = get_provider("gemini", use_vertex=False)
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "Google AI Studio (Gemini)"
    assert provider.supports_video is True

def test_get_provider_gemini_vertex(monkeypatch):
    # Pour Vertex, on fake un faux ID de projet
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "mon-faux-projet")
    
    # On "mock" genai.Client pour éviter l'erreur de credentials par défaut sur Vertex
    import google.genai as genai
    class DummyClient:
        def __init__(self, **kwargs):
            pass
    monkeypatch.setattr(genai, "Client", DummyClient)
    
    provider = get_provider("gemini", use_vertex=True, project_id="dummy-project")
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "Vertex AI (Gemini)"
    assert provider.supports_video is True

def test_get_provider_openai():
    provider = get_provider("openai", api_key="dummy")
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "OpenAI"
    assert provider.supports_video is False

def test_get_provider_unsupported():
    with pytest.raises(ValueError, match="Provider non supporté: unknown"):
        get_provider("unknown")

def test_llm_message_creation():
    msg = LLMMessage(parts=[
        TextPart(text="hello"),
        MediaPart(media=b"dummy_bytes", mime_type="image/jpeg"),
        MediaPart(media="http://example.com/img.png"),
    ])
    assert msg.text == "hello"
    assert len(msg.media) == 2

def test_llm_call_config_defaults():
    config = LLMCallConfig(model_name="gpt-4o")
    assert config.temperature == 0.0
    assert config.max_tokens is None
    assert config.timeout == 30
