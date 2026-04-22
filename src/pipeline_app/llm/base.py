from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Any

class LLMMessage(BaseModel):
    text: str
    media: list[str | bytes] = Field(default_factory=list)

class LLMCallConfig(BaseModel):
    model_name: str
    temperature: float = 0.0 
    max_tokens: int | None = None
    timeout: int = 30 

class LLMResponse(BaseModel):
    parsed: BaseModel 
    raw_json: str
    usage: tuple[int, int] 
    latency_ms: int
    model_used: str

class TraceContext:
    """Conteneur léger pour propager le contexte Langfuse (Span)."""
    def __init__(self, span: Any = None):
        self.span = span

class LLMProvider(ABC):
    """Interface abstraite pour tous les fournisseurs d'IA."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def supports_video(self) -> bool:
        pass

    @abstractmethod
    def generate_structured(
        self, 
        messages: list[LLMMessage], 
        response_model: type[BaseModel], 
        config: LLMCallConfig, 
        trace_context: TraceContext | None = None
    ) -> LLMResponse:
        """Génère une réponse structurée à partir de messages et de médias."""
        pass