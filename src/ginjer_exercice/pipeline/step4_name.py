"""Step 4 — Explicit product name extraction.

Extrait le nom d'un produit UNIQUEMENT s'il est explicitement mentionné
dans le contenu de la pub (texte ou visible dans les médias). Si non trouvé,
retourne ``None`` pour signaler à l'orchestrator qu'il faut déclencher le
fallback (Step 5).

C'est l'étape la plus délicate : le LLM est naturellement tenté d'inventer
un nom plausible basé sur sa connaissance générale de la marque. Le prompt
insiste lourdement sur l'interdiction de deviner.

Usage::

    from ginjer_exercice.pipeline import step4_name

    name = step4_name.execute(
        product,
        classification,
        ad,
        llm_provider=llm,
        prompt_registry=registry,
        trace=trace,
    )
    if name is None:
        # Déclencher step5 (fallback)
        pass
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm.base import LLMCallConfig, LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.tracing import step_span
from ..schemas.ad import Ad
from ..schemas.products import DetectedProduct, ProductClassification, ProductName
from ..schemas.step_outputs import ExtractedName
from ._helpers import build_llm_messages, build_texts_block

logger = logging.getLogger(__name__)

_PROMPT_NAME = "pipeline/name_extraction"


def execute(
    product: DetectedProduct,
    classification: ProductClassification,
    ad: Ad,
    *,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    trace: Any,
) -> ProductName | None:
    """Extrait le nom explicite d'un produit depuis le contenu de la pub.

    Le LLM ne doit pas deviner : si le nom n'est pas dans le texte ou
    visible dans les médias, il doit retourner ``name: null``.
    Cette règle est renforcée dans le prompt avec des exemples négatifs.

    Retourner ``None`` signale à l'orchestrator (Phase 6) qu'il faut
    déclencher le Step 5 (fallback web search + catalogue).

    Args:
        product: Le produit à nommer (description visuelle).
        classification: La classification taxonomique du produit (step3).
        ad: La publicité source (textes + médias).
        llm_provider: Fournisseur LLM.
        prompt_registry: Registre des prompts versionnés.
        trace: Contexte de trace Langfuse (span parent).

    Returns:
        ``ProductName`` si le nom est explicitement trouvé dans le contenu,
        ``None`` si le LLM ne trouve pas de nom explicite (fallback requis).

    Raises:
        ValueError: Si le LLM échoue à renvoyer un JSON valide après retries.
        FileNotFoundError: Si le prompt est introuvable.
    """
    with step_span(
        name="step_4_name",
        input_payload={
            "product": product.model_dump(),
            "classification": classification.model_dump(),
            "ad_id": ad.platform_ad_id,
            "brand": ad.brand.value,
        },
    ) as span:
        # 1. Récupérer le prompt
        prompt = prompt_registry.get(_PROMPT_NAME)

        # 2. Compiler le prompt
        texts_block = build_texts_block(ad.texts)
        compiled = prompt.compile(
            brand=ad.brand.value,
            product_description=product.raw_description,
            product_universe=classification.universe,
            product_category=classification.category,
            product_subcategory=classification.subcategory,
            texts_block=texts_block,
            media_count=str(len(ad.media_urls)),
        )

        # 3. Construire les messages multimodaux
        messages = build_llm_messages(compiled, ad.media_urls)

        # 4. Configurer l'appel LLM
        llm_config = LLMCallConfig(
            model_name=prompt.config.get("model", "gemini-2.0-flash"),
            temperature=prompt.config.get("temperature", 0.0),
            max_tokens=prompt.config.get("max_tokens", 1000),
        )

        # 5. Appel LLM
        logger.debug(
            "Step4 — Extraction de nom pour ad_id=%s, produit=%s",
            ad.platform_ad_id,
            product.raw_description[:60],
        )
        response = llm_provider.generate_structured(
            messages=messages,
            response_model=ExtractedName,
            config=llm_config,
            trace_context=trace,
        )
        extracted: ExtractedName = response.parsed  # type: ignore[assignment]

        # 6. Conversion vers ProductName (ou None si pas de nom)
        result = _to_product_name(extracted)

        if result is None:
            logger.info(
                "Step4 — Aucun nom explicite trouvé pour ad_id=%s → fallback requis.",
                ad.platform_ad_id,
            )
        else:
            logger.info(
                "Step4 — Nom extrait pour ad_id=%s : '%s' (found_in=%s, conf=%.2f)",
                ad.platform_ad_id,
                result.name,
                extracted.found_in,
                extracted.confidence,
            )

        span.update_output(result.model_dump() if result else None)
        return result


def _to_product_name(extracted: ExtractedName) -> ProductName | None:
    """Convertit un ``ExtractedName`` (sortie LLM) en ``ProductName`` (schéma final).

    Retourne ``None`` si le LLM a indiqué qu'il n'a pas trouvé de nom
    (``name is None`` ou ``found_in == "none"``).

    Args:
        extracted: Sortie brute du LLM.

    Returns:
        ``ProductName`` si un nom a été trouvé, ``None`` sinon.
    """
    if extracted.name is None or extracted.found_in == "none":
        return None

    return ProductName(
        name=extracted.name,
        source="explicit",
        confidence=extracted.confidence,
        needs_review=False,  # Un nom explicite ne nécessite pas de review
        sources_consulted=[f"ad_{extracted.found_in}"],
    )
