"""Fakes et helpers réutilisables pour les tests du pipeline.

Ces classes sont des doubles de test (Test Doubles) qui implémentent
les interfaces de production sans appel réseau ni dépendance externe.

Usage dans les tests::

    from tests.unit.pipeline_fakes import FakeLLMProvider, FakePromptRegistry

    fake_llm = FakeLLMProvider(canned_responses=[universe_result])
    fake_registry = FakePromptRegistry()
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel

from ginjer_exercice.llm.base import (
    LLMCallConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    TraceContext,
)
from ginjer_exercice.observability.prompts import ManagedPrompt, PromptRegistry


class FakeLLMProvider(LLMProvider):
    """LLM provider qui retourne des réponses prédéfinies dans l'ordre des appels.

    Chaque appel à ``generate_structured`` consomme la première réponse
    de la file. Si la file est vide, lève une assertion.

    Usage::

        fake = FakeLLMProvider([UniverseResult(detected_universes=[...])])
        response = fake.generate_structured(messages, UniverseResult, config)
        assert response.parsed == fake.response[0]
    """

    def __init__(self, canned_responses: list[BaseModel]):
        self._responses: deque[BaseModel] = deque(canned_responses)
        self.calls: list[dict[str, Any]] = []  # Historique des appels

    @property
    def name(self) -> str:
        return "FakeLLMProvider"

    @property
    def supports_video(self) -> bool:
        return True

    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        config: LLMCallConfig,
        trace_context: TraceContext | None = None,
    ) -> LLMResponse:
        assert self._responses, (
            f"FakeLLMProvider: la file de réponses est vide. "
            f"Appels effectués : {len(self.calls)}. "
            f"Vérifiez que vous avez fourni assez de canned_responses."
        )
        self.calls.append({
            "messages": messages,
            "response_model": response_model,
            "config": config,
        })
        parsed = self._responses.popleft()
        return LLMResponse(
            parsed=parsed,
            raw_json=parsed.model_dump_json(),
            usage=(100, 50),
            latency_ms=42,
            model_used="fake-model",
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


class SequentialFakeLLMProvider(FakeLLMProvider):
    """Variante de FakeLLMProvider qui permet de cycler sur les réponses.

    Utile pour les tests de retry où on veut simuler N réponses invalides
    puis une réponse valide.
    """

    def __init__(self, canned_responses: list[BaseModel], cycle: bool = False):
        super().__init__(canned_responses)
        self._all_responses = list(canned_responses)
        self._cycle = cycle
        self._call_index = 0

    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        config: LLMCallConfig,
        trace_context: TraceContext | None = None,
    ) -> LLMResponse:
        if not self._responses and self._cycle:
            self._responses = deque(self._all_responses)
        return super().generate_structured(messages, response_model, config, trace_context)


class FakePromptRegistry:
    """Registre de prompts qui retourne des prompts statiques pour les tests.

    Retourne toujours un ``ManagedPrompt`` minimal avec un template neutre
    qui passe les variables sans transformation.
    """

    def __init__(self, prompt_override: str | None = None):
        # Template neutre qui expose les variables telles quelles
        self._prompt_text = prompt_override or (
            "Brand: {{brand}}\n"
            "Text: {{texts_block}}\n"
            "Media: {{media_count}}\n"
            "Universes: {{universes}}\n"
            "Product: {{product_description}}\n"
            "Color: {{product_color}}\n"
            "Universe: {{product_universe}}\n"
            "Category: {{product_category}}\n"
            "Subcategory: {{product_subcategory}}\n"
            "Taxonomy: {{taxonomy_tree}}\n"
        )
        self.gets: list[str] = []

    def get(self, name: str, label: str = "production") -> ManagedPrompt:
        self.gets.append(name)
        return ManagedPrompt(
            name=name,
            version="test-v1",
            label=label,
            prompt=self._prompt_text,
            config={
                "model": "fake-model",
                "temperature": 0.0,
                "max_tokens": 100,
            },
            source="yaml_fallback",
        )


class FakeTraceSpan:
    """Fake pour les spans Langfuse — enregistre les appels update() pour assertion."""

    def __init__(self, name: str = "fake-span"):
        self.name = name
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def __enter__(self) -> "FakeTraceSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
