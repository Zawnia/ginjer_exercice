import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from .base import LLMCallConfig, LLMMessage, LLMProvider, LLMResponse, MediaPart, TextPart, TraceContext
from ..observability.runtime_warnings import add_runtime_warning

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self, use_vertex: bool = False, project_id: str | None = None, location: str = "us-central1"):
        self.use_vertex = use_vertex

        if self.use_vertex:
            if not project_id:
                raise ValueError("project_id est requis pour Vertex AI")
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
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
        """Convertit une URL publique GCS en URI `gs://` acceptée par Gemini."""
        prefix = "https://storage.googleapis.com/"
        if url.startswith(prefix):
            return f"gs://{url[len(prefix):]}"
        return url

    def _build_gemini_contents(self, messages: list[LLMMessage]) -> list[types.Content]:
        """Convertit les messages génériques en types natifs Gemini."""
        contents: list[types.Content] = []
        for msg in messages:
            parts: list[types.Part] = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    if part.text:
                        parts.append(types.Part.from_text(text=part.text))
                    continue

                if isinstance(part, MediaPart):
                    media = part.media
                    mime_type = part.mime_type
                    if isinstance(media, bytes):
                        parts.append(types.Part.from_bytes(data=media, mime_type=mime_type or "image/jpeg"))
                    elif isinstance(media, str):
                        gs_uri = media if media.startswith("gs://") else self._convert_http_to_gs_uri(media)
                        if gs_uri.startswith("gs://"):
                            resolved_mime = mime_type or ("video/mp4" if media.endswith((".mp4", ".mov")) else "image/jpeg")
                            parts.append(types.Part.from_uri(file_uri=gs_uri, mime_type=resolved_mime))
                        else:
                            raise ValueError(
                                "URL de média non supportée directement par Gemini "
                                f"(doit être gs:// ou téléchargeable) : {media}"
                            )

            contents.append(types.Content(role="user", parts=parts))
        return contents

    def _extract_usage(self, response: Any) -> tuple[int, int]:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            return (
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0,
            )
        return (0, 0)

    def _response_diagnostics(self, response: Any) -> dict[str, Any]:
        usage = self._extract_usage(response)
        diagnostics: dict[str, Any] = {
            "text_length": len(getattr(response, "text", "") or ""),
            "prompt_tokens": usage[0],
            "output_tokens": usage[1],
        }
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reasons = []
            for candidate in candidates:
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason is not None:
                    finish_reasons.append(str(finish_reason))
            if finish_reasons:
                diagnostics["finish_reasons"] = finish_reasons
        return diagnostics

    def _build_generation_config(
        self,
        *,
        response_model: type[BaseModel],
        config: LLMCallConfig,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            response_mime_type="application/json",
            response_schema=response_model,
        )

    def _build_repair_contents(
        self,
        *,
        response_model: type[BaseModel],
        invalid_json: str,
        validation_error: str,
    ) -> list[types.Content]:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        repair_prompt = (
            "You must repair an invalid JSON response.\n"
            "Return only valid JSON matching the target schema exactly.\n"
            f"Validation error: {validation_error}\n"
            f"Target JSON schema: {schema}\n"
            "Invalid JSON to repair:\n"
            f"{invalid_json}"
        )
        return [types.Content(role="user", parts=[types.Part.from_text(text=repair_prompt)])]

    def _parse_response_payload(
        self,
        *,
        response: Any,
        response_model: type[BaseModel],
    ) -> tuple[BaseModel, str]:
        parsed_attr = getattr(response, "parsed", None)
        if parsed_attr is not None:
            if isinstance(parsed_attr, response_model):
                raw_json = parsed_attr.model_dump_json()
                return parsed_attr, raw_json
            parsed_obj = response_model.model_validate(parsed_attr)
            raw_json = parsed_obj.model_dump_json()
            return parsed_obj, raw_json

        text_response = response.text or "{}"
        parsed_obj = response_model.model_validate_json(text_response)
        return parsed_obj, text_response

    def _log_generation(
        self,
        *,
        trace_context: TraceContext | None,
        config: LLMCallConfig,
        messages: list[LLMMessage],
        parsed_obj: BaseModel,
        usage: tuple[int, int],
    ) -> None:
        if trace_context is None:
            return
        trace_context.log_generation(
            name=f"llm_{config.model_name}",
            model=config.model_name,
            input=[m.model_dump() for m in messages],
            output=parsed_obj.model_dump(),
            usage_details={
                "input_tokens": usage[0],
                "output_tokens": usage[1],
            },
            model_parameters={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
        )

    def _repair_invalid_json(
        self,
        *,
        response_model: type[BaseModel],
        config: LLMCallConfig,
        invalid_json: str,
        validation_error: str,
    ) -> tuple[BaseModel, str, tuple[int, int], int]:
        repair_contents = self._build_repair_contents(
            response_model=response_model,
            invalid_json=invalid_json,
            validation_error=validation_error,
        )
        generation_config = self._build_generation_config(response_model=response_model, config=config)
        start_time = time.time()
        response = self.client.models.generate_content(
            model=config.model_name,
            contents=repair_contents,
            config=generation_config,
        )
        latency = int((time.time() - start_time) * 1000)
        parsed_obj, raw_json = self._parse_response_payload(response=response, response_model=response_model)
        usage = self._extract_usage(response)
        return parsed_obj, raw_json, usage, latency

    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        config: LLMCallConfig,
        trace_context: TraceContext | None = None,
    ) -> LLMResponse:
        """Génère une réponse structurée avec tentative de réparation sur JSON invalide."""
        contents = self._build_gemini_contents(messages)
        generation_config = self._build_generation_config(response_model=response_model, config=config)

        max_retries = 2
        last_error: str | None = None

        for attempt in range(max_retries + 1):
            start_time = time.time()
            response = None
            try:
                response = self.client.models.generate_content(
                    model=config.model_name,
                    contents=contents,
                    config=generation_config,
                )
                latency = int((time.time() - start_time) * 1000)
                parsed_obj, raw_json = self._parse_response_payload(
                    response=response,
                    response_model=response_model,
                )
                usage = self._extract_usage(response)
                self._log_generation(
                    trace_context=trace_context,
                    config=config,
                    messages=messages,
                    parsed_obj=parsed_obj,
                    usage=usage,
                )
                return LLMResponse(
                    parsed=parsed_obj,
                    raw_json=raw_json,
                    usage=usage,
                    latency_ms=latency,
                    model_used=config.model_name,
                )

            except ValidationError as exc:
                last_error = str(exc)
                invalid_json = response.text if response is not None and getattr(response, "text", None) else "{}"
                logger.warning(
                    "Gemini structured output validation failed, retrying (%d/%d): %s | diagnostics=%s",
                    attempt + 1,
                    max_retries + 1,
                    last_error,
                    self._response_diagnostics(response),
                )
                add_runtime_warning("gemini: repaired malformed JSON response")

                try:
                    repaired_obj, repaired_raw_json, repaired_usage, repaired_latency = self._repair_invalid_json(
                        response_model=response_model,
                        config=config,
                        invalid_json=invalid_json,
                        validation_error=last_error,
                    )
                    logger.info(
                        "Gemini invalid JSON repaired successfully for %s on attempt %d",
                        response_model.__name__,
                        attempt + 1,
                    )
                    self._log_generation(
                        trace_context=trace_context,
                        config=config,
                        messages=messages,
                        parsed_obj=repaired_obj,
                        usage=repaired_usage,
                    )
                    return LLMResponse(
                        parsed=repaired_obj,
                        raw_json=repaired_raw_json,
                        usage=repaired_usage,
                        latency_ms=repaired_latency,
                        model_used=config.model_name,
                    )
                except ValidationError as repair_exc:
                    last_error = str(repair_exc)
                    logger.warning(
                        "Gemini JSON repair attempt failed for %s (%d/%d): %s",
                        response_model.__name__,
                        attempt + 1,
                        max_retries + 1,
                        last_error,
                    )

        raise ValueError(
            "Échec de génération d'un output structuré valide après "
            f"{max_retries} retries. Dernière erreur : {last_error}"
        )
