"""Step 1 — Universe detection.

Détecte les univers produits représentés dans une publicité (Fashion, Beauty, etc.).
C'est la première étape du pipeline et elle sert de template pour les autres.

Usage::

    from ginjer_exercice.pipeline import step1_universe
    from ginjer_exercice.observability.tracing import pipeline_trace

    with pipeline_trace(ad=ad) as trace:
        result = step1_universe.execute(
            ad,
            llm_provider=llm,
            prompt_registry=registry,
            trace=trace,
        )
    # result.detected_universes → liste des univers avec confiance
    # result.primary_universe → univers principal
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm.base import LLMCallConfig, LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.tracing import step_span
from ..schemas.ad import Ad
from ..schemas.step_outputs import UniverseResult
from ._helpers import build_llm_messages, build_texts_block

logger = logging.getLogger(__name__)

_PROMPT_NAME = "pipeline/universe"


def execute(
    ad: Ad,
    *,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    trace: Any,
) -> UniverseResult:
    """Détecte les univers produits dans une publicité.

    Analyse le contenu textuel et visuel (images/vidéos) d'une pub pour
    déterminer quels univers (Women, Beauty, Fragrance, etc.) sont représentés.

    Une pub peut présenter plusieurs univers simultanément (ex : un modèle
    portant un sac dans une pub parfum → Women + Fragrance).

    Args:
        ad: La publicité à analyser. Doit contenir au moins ``texts`` ou
            ``media_urls`` pour produire un résultat utile.
        llm_provider: Fournisseur LLM (Gemini, OpenAI, etc.).
        prompt_registry: Registre des prompts versionnés (Langfuse + YAML fallback).
        trace: Contexte de trace Langfuse (span parent). Peut être un NullSpanContext.

    Returns:
        ``UniverseResult`` avec la liste des univers détectés, triés par confiance.

    Raises:
        ValueError: Si le LLM échoue à renvoyer un JSON valide après retries
            (levée par le provider sous-jacent).
        FileNotFoundError: Si le prompt ``pipeline/universe`` est absent de Langfuse
            et du fallback YAML local.
    """
    with step_span(name="step_1_universe", input_payload=ad.model_dump()) as span:
        # 1. Récupérer le prompt versionné
        prompt = prompt_registry.get(_PROMPT_NAME)

        # 2. Compiler le prompt avec les variables dynamiques
        texts_block = build_texts_block(ad.texts)
        compiled = prompt.compile(
            brand=ad.brand.value,
            texts_block=texts_block,
            media_count=str(len(ad.media_urls)),
        )

        # 3. Construire les messages multimodaux (texte + médias)
        messages = build_llm_messages(compiled, ad.media_urls)

        # 4. Configurer l'appel LLM depuis le config du prompt
        llm_config = LLMCallConfig(
            model_name=prompt.config.get("model", "gemini-2.0-flash"),
            temperature=prompt.config.get("temperature", 0.1),
            max_tokens=prompt.config.get("max_tokens", 2000),
        )

        # 5. Appel LLM avec schema Pydantic
        logger.debug(
            "Step1 — Appel LLM pour ad_id=%s, brand=%s, media_count=%d",
            ad.platform_ad_id,
            ad.brand.value,
            len(ad.media_urls),
        )
        response = llm_provider.generate_structured(
            messages=messages,
            response_model=UniverseResult,
            config=llm_config,
            trace_context=trace,
        )

        result: UniverseResult = response.parsed  # type: ignore[assignment]

        # 6. Post-traitement : trier par confiance décroissante
        result = UniverseResult(
            detected_universes=sorted(
                result.detected_universes,
                key=lambda u: u.confidence,
                reverse=True,
            )
        )

        logger.info(
            "Step1 — ad_id=%s → %d univers detectes : %s",
            ad.platform_ad_id,
            len(result.detected_universes),
            [u.universe for u in result.detected_universes],
        )

        span.update_output(result.model_dump())
        return result
