"""Step 2 - Product detection."""

from __future__ import annotations

import logging
from typing import Any

from ..data_access.media_fetcher import MediaFetcher
from ..llm.base import LLMCallConfig, LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.runtime_warnings import add_runtime_warning
from ..observability.tracing import step_span
from ..schemas.ad import Ad
from ..schemas.media import MediaKind
from ..schemas.products import Color, DetectedProduct
from ..schemas.step_outputs import DetectedProductList, UniverseResult
from ._helpers import build_llm_messages, build_message_with_media, build_texts_block, select_media_urls

logger = logging.getLogger(__name__)

_PROMPT_NAME = "pipeline/products"
IMPORTANCE_THRESHOLD = 0
_WARNING_TEXT_ONLY = "step2: fallback to text-only, no usable image"
_WARNING_VIDEO_IGNORED = "step2: video ignored (not supported in P0)"


def execute(
    ad: Ad,
    universe_result: UniverseResult,
    *,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    trace: Any,
    media_fetcher: MediaFetcher,
) -> list[DetectedProduct]:
    """Identifie tous les produits presents dans une publicite."""
    with step_span(
        name="step_2_products",
        input_payload={
            "ad_id": ad.platform_ad_id,
            "brand": ad.brand.value,
            "universes": universe_result.universe_names,
        },
    ) as span:
        prompt = prompt_registry.get(_PROMPT_NAME)

        texts_block = build_texts_block(ad.texts)
        universes_str = ", ".join(universe_result.universe_names) or "Unknown"

        compiled = prompt.compile(
            brand=ad.brand.value,
            texts_block=texts_block,
            universes=universes_str,
            media_count=str(len(ad.media_urls)),
        )

        messages = _build_step2_messages(
            compiled,
            ad.media_urls,
            media_fetcher=media_fetcher,
        )

        llm_config = LLMCallConfig(
            model_name=prompt.config.get("model", "gemini-2.0-flash"),
            temperature=prompt.config.get("temperature", 0.1),
            max_tokens=prompt.config.get("max_tokens", 4000),
        )

        logger.debug(
            "Step2 - Appel LLM pour ad_id=%s, universes=%s",
            ad.platform_ad_id,
            universes_str,
        )
        response = llm_provider.generate_structured(
            messages=messages,
            response_model=DetectedProductList,
            config=llm_config,
            trace_context=trace,
        )
        llm_output: DetectedProductList = response.parsed  # type: ignore[assignment]

        products = _convert_and_filter(llm_output, ad.platform_ad_id)

        logger.info(
            "Step2 - ad_id=%s -> %d produits detectes (avant filtre: %d), overall_confidence=%.2f",
            ad.platform_ad_id,
            len(products),
            len(llm_output.products),
            llm_output.overall_confidence,
        )

        span.update_output([p.model_dump() for p in products])
        return products


def _build_step2_messages(
    prompt_text: str,
    media_urls: list[str],
    *,
    media_fetcher: MediaFetcher,
) -> list:
    selected_urls = select_media_urls(media_urls)
    if not selected_urls:
        return build_llm_messages(prompt_text, [])

    downloaded_images = []
    video_warning_added = False

    for url in selected_urls:
        if not url.startswith(("http://", "https://")):
            logger.info("Step2 - media ignored because URL is not downloadable by MediaFetcher: %s", url)
            continue

        try:
            media = media_fetcher.download(url)
        except Exception as exc:
            logger.warning("Step2 - media download failed for %s: %s", url, exc)
            continue

        if media.kind != MediaKind.IMAGE:
            logger.info("Step2 - video ignored in P0 for %s", url)
            if not video_warning_added:
                add_runtime_warning(_WARNING_VIDEO_IGNORED)
                video_warning_added = True
            continue

        downloaded_images.append(media)

    if not downloaded_images:
        logger.info("Step2 - no usable image media, falling back to text-only.")
        add_runtime_warning(_WARNING_TEXT_ONLY)
        return build_llm_messages(prompt_text, [])

    return build_message_with_media(prompt_text, downloaded_images)


def _convert_and_filter(
    llm_output: DetectedProductList,
    ad_id: str,
) -> list[DetectedProduct]:
    """Convertit les sorties LLM en DetectedProduct et filtre par importance."""
    products: list[DetectedProduct] = []

    for raw in llm_output.products:
        if raw.importance <= IMPORTANCE_THRESHOLD:
            logger.debug(
                "Step2 - Produit filtre (importance=%d) : %s",
                raw.importance,
                raw.raw_description[:60],
            )
            continue

        color = _normalize_color(raw.color, ad_id)

        products.append(
            DetectedProduct(
                raw_description=raw.raw_description,
                universe=raw.universe,
                color=color,
                importance=raw.importance,
            )
        )

    return sorted(products, key=lambda p: p.importance, reverse=True)


def _normalize_color(raw_color: str, ad_id: str) -> Color:
    """Normalise une couleur string LLM vers l'enum Color."""
    try:
        return Color(raw_color)
    except ValueError:
        pass

    raw_lower = raw_color.strip().lower()
    for color in Color:
        if color.value.lower() == raw_lower:
            return color

    logger.warning(
        "Step2 - Couleur LLM non reconnue '%s' pour ad_id=%s -> Multicolor.",
        raw_color,
        ad_id,
    )
    return Color.MULTICOLOR
