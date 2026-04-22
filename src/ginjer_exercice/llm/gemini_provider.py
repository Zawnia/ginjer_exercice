from ginjer_exercice.llm.base import LLMProvider, LLMCallConfig, LLMMessage, LLMResponse, TraceContext
from pydantic import BaseModel
from google import genai

class GeminiProvider(LLMProvider):
    def __init__(self, use_vertex: bool = False, project_id: str | None = None, location: str = "us-central1"):
        self.use_vertex = use_vertex
        
        if self.use_vertex:
            if not project_id:
                raise ValueError("project_id est requis pour Vertex AI")
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=location
            )
        else:
            self.client = genai.Client()

    @property
    def name(self) -> str:
        return "Vertex AI (Gemini)" if self.use_vertex else "Google AI Studio (Gemini)"

    @property
    def supports_video(self) -> bool:
        return True

    def generate_structured(self, messages: list[LLMMessage], response_model: type[BaseModel], config: LLMCallConfig, trace_context: TraceContext | None = None) -> LLMResponse:
        
        pass