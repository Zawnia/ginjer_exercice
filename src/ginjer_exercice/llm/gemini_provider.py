import os
import time
from typing import Any
import json
from pydantic import BaseModel, ValidationError

from google import genai
from google.genai import types

from .base import LLMCallConfig, LLMMessage, LLMProvider, LLMResponse, TraceContext


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

    def _convert_http_to_gs_uri(self, url: str) -> str:
        """
        Convertit une URL publique GCP (https://storage.googleapis.com/bucket/path)
        en URI native GCS (gs://bucket/path) acceptée par l'API Gemini.
        """
        prefix = "https://storage.googleapis.com/"
        if url.startswith(prefix):
            return f"gs://{url[len(prefix):]}"

        return url

    def _build_gemini_contents(self, messages: list[LLMMessage]) -> list[types.Content]:
        """Convertit les messages génériques en types natifs Gemini."""
        contents = []
        for msg in messages:
            parts = []
            if msg.text:
                parts.append(types.Part.from_text(text=msg.text))
            
            for media in msg.media:
                if isinstance(media, bytes):
                    parts.append(types.Part.from_bytes(data=media, mime_type="image/jpeg"))
                elif isinstance(media, str):
                    gs_uri = media if media.startswith("gs://") else self._convert_http_to_gs_uri(media)
                    
                    if gs_uri.startswith("gs://"):
                        mime_type = "video/mp4" if media.endswith((".mp4", ".mov")) else "image/jpeg"
                        parts.append(types.Part.from_uri(file_uri=gs_uri, mime_type=mime_type))
                    else:
                        raise ValueError(f"URL de média non supportée directement par Gemini (doit être gs:// ou téléchargeable) : {media}")
            
            contents.append(types.Content(role="user", parts=parts))
        return contents

    def generate_structured(
        self, 
        messages: list[LLMMessage], 
        response_model: type[BaseModel], 
        config: LLMCallConfig, 
        trace_context: TraceContext | None = None
    ) -> LLMResponse:
        """Génère une réponse structurée avec retry sur erreur de parsing (LLM-Healing)."""
        
        contents = self._build_gemini_contents(messages)
        
        generation_config = types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            response_mime_type="application/json",
            response_schema=response_model,
        )

        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            start_time = time.time()
            
            try:
                # 1. Appel à l'API
                response = self.client.models.generate_content(
                    model=config.model_name,
                    contents=contents,
                    config=generation_config
                )
                latency = int((time.time() - start_time) * 1000)
                
                text_response = response.text or "{}"
                
                parsed_obj = response_model.model_validate_json(text_response)
                
                usage = (0, 0)
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = (
                        response.usage_metadata.prompt_token_count or 0, 
                        response.usage_metadata.candidates_token_count or 0
                    )

                return LLMResponse(
                    parsed=parsed_obj,
                    raw_json=text_response,
                    usage=usage,
                    latency_ms=latency,
                    model_used=config.model_name
                )
                
            except ValidationError as e:
                last_error = str(e)
                error_message = (
                    "Le JSON précédent était invalide ou ne respectait pas le schéma strict demandé.\n"
                    f"Voici l'erreur renvoyée par le validateur : {last_error}\n"
                    "Corrige ta réponse pour qu'elle respecte EXACTEMENT le format attendu."
                )
                
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=response.text)]))
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=error_message)]))
                
                print(f"[GeminiProvider] ValidationError (Tentative {attempt+1}/{max_retries+1}) : Retry initié.")
        
        raise ValueError(f"Échec de génération d'un output structuré valide après {max_retries} retries. Dernière erreur : {last_error}")