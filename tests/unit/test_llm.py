from ginjer_exercice.llm.factory import get_provider
from ginjer_exercice.llm.base import LLMProvider
from ginjer_exercice.llm.gemini_provider import GeminiProvider
from ginjer_exercice.llm.openai_provider import OpenAIProvider
from ginjer_exercice.llm.base import LLMMessage, LLMCallConfig
from pydantic import BaseModel
import pytest

class DummyResponseModel(BaseModel):
    name: str
    confidence: float

def test_get_provider_gemini():
    provider = get_provider("gemini", use_vertex=False)
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "Google AI Studio (Gemini)"
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
    msg = LLMMessage(text="hello", media=[b"dummy_bytes", "http://example.com/img.png"])
    assert msg.text == "hello"
    assert len(msg.media) == 2

def test_llm_call_config_defaults():
    config = LLMCallConfig(model_name="gpt-4o")
    assert config.temperature == 0.0
    assert config.max_tokens is None
    assert config.timeout == 30
