import base64
import time

from openai import OpenAI
from pydantic import BaseModel

from .base import LLMCallConfig, LLMMessage, LLMProvider, LLMResponse, TraceContext


class OpenAIProvider(LLMProvider):
    
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "OpenAI"

    @property
    def supports_video(self) -> bool:
        return False

    def generate_structured(
        self, 
        messages: list[LLMMessage], 
        response_model: type[BaseModel], 
        config: LLMCallConfig, 
        trace_context: TraceContext | None = None
    ) -> LLMResponse:
        """Génère une réponse structurée en utilisant l'API OpenAI."""
        
        oai_messages = []
        for msg in messages:
            content = []
            if msg.text:
                content.append({"type": "text", "text": msg.text})
            for media in msg.media:
                if isinstance(media, bytes):
                    b64_img = base64.b64encode(media).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    })
                elif isinstance(media, str) and (media.startswith("http://") or media.startswith("https://")):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": media}
                    })
            oai_messages.append({"role": "user", "content": content})

        start_time = time.time()
        
        response = self.client.beta.chat.completions.parse(
            model=config.model_name,
            messages=oai_messages,
            response_format=response_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
        
        latency = int((time.time() - start_time) * 1000)
        
        message = response.choices[0].message
        
        usage = (0, 0)
        if response.usage:
            usage = (response.usage.prompt_tokens, response.usage.completion_tokens)
            
        return LLMResponse(
            parsed=message.parsed,
            raw_json=message.content or "{}",
            usage=usage,
            latency_ms=latency,
            model_used=response.model
        )
