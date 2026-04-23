"""Step 2 — Product detection.

Identifie tous les produits présents dans une publicité, en s'appuyant sur
les univers détectés à l'étape 1 pour contextualiser la recherche.

Usage::

    from ginjer_exercice.pipeline import step2_products

    products = step2_products.execute(
        ad,
        universe_result,
        llm_provider=llm,
        prompt_registry=registry,
        trace=trace,
    )
    # products → liste de DetectedProduct (importance > 0 recommandé)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import get_settings
from ..data_access.media_fetcher import MediaFetcher
from ..llm.base import LLMCallConfig, LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.tracing import step_span
from ..schemas.ad import Ad
from ..schemas.media import MediaKind
from ..schemas.products import Color, DetectedProduct
from ..schemas.step_outputs import DetectedProductList, UniverseResult
from ._helpers import build_llm_messages, build_message_with_media, build_texts_block, select_media_urls

logger = logging.getLogger(__name__)

_PROMPT_NAME = "pipeline/products"

# Seuil d'importance : les produits avec importance == 0 sont filtrés avant step3
# pour éviter de gaspiller des tokens LLM sur des produits que le LLM lui-même
# n'est pas sûr de voir. Ce filtre est documenté ici pour traçabilité.
IMPORTANCE_THRESHOLD = 0


def execute(
    ad: Ad,
    universe_result: UniverseResult,
    *,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    trace: Any,
    media_fetcher: MediaFetcher | None = None,
) -> list[DetectedProduct]:
    """Identifie tous les produits présents dans une publicité.

    S'appuie sur les univers détectés à l'étape 1 pour contextualiser
    le prompt. Les produits avec importance == 0 sont filtrés — ils
    indiquent que le LLM lui-même n'est pas sûr de les voir, et il est
    inutile de les classifier.

    Args:
        ad: La publicité à analyser.
        universe_result: Les univers détectés par step1.
        llm_provider: Fournisseur LLM.
        prompt_registry: Registre des prompts versionnés.
        trace: Contexte de trace Langfuse (span parent).

    Returns:
        Liste de ``DetectedProduct`` avec importance > ``IMPORTANCE_THRESHOLD``,
        triée par importance décroissante (produits principaux en premier).
        Peut être une liste vide si aucun produit n'est identifiable.

    Raises:
        ValueError: Si le LLM échoue à renvoyer un JSON valide après retries.
        FileNotFoundError: Si le prompt est introuvable.
    """
    with step_span(
        name="step_2_products",
        input_payload={
            "ad_id": ad.platform_ad_id,
            "brand": ad.brand.value,
            "universes": universe_result.universe_names,
        },
    ) as span:
        # 1. Récupérer le prompt
        prompt = prompt_registry.get(_PROMPT_NAME)

        # 2. Compiler avec les variables dynamiques
        texts_block = build_texts_block(ad.texts)
        universes_str = ", ".join(universe_result.universe_names) or "Unknown"

        compiled = prompt.compile(
            brand=ad.brand.value,
            texts_block=texts_block,
            universes=universes_str,
            media_count=str(len(ad.media_urls)),
        )

        # 3. Construire les messages multimodaux
        messages = _build_step2_messages(
            compiled,
            ad.media_urls,
            media_fetcher=media_fetcher,
        )

        # 4. Configurer l'appel LLM
        llm_config = LLMCallConfig(
            model_name=prompt.config.get("model", "gemini-2.0-flash"),
            temperature=prompt.config.get("temperature", 0.1),
            max_tokens=prompt.config.get("max_tokens", 4000),
        )

        # 5. Appel LLM
        logger.debug(
            "Step2 — Appel LLM pour ad_id=%s, universes=%s",
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

        # 6. Post-traitement : conversion + filtre importance
        products = _convert_and_filter(llm_output, ad.platform_ad_id)

        logger.info(
            "Step2 — ad_id=%s → %d produits detectes (avant filtre: %d), "
            "overall_confidence=%.2f",
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
    media_fetcher: MediaFetcher | None,
) -> list:
    selected_urls = select_media_urls(media_urls)
    if not selected_urls:
        return build_llm_messages(prompt_text, [])

    fetcher = media_fetcher or _default_media_fetcher()
    downloaded_images = []
    for url in selected_urls:
        if not url.startswith(("http://", "https://")):
            logger.info("Step2 — média ignoré car URL non téléchargeable par MediaFetcher : %s", url)
            continue

        media = None
        try:
            media = fetcher.download(url)
        except Exception as exc:
            logger.warning("Step2 — téléchargement média échoué pour %s : %s", url, exc)
            continue

        if media.kind != MediaKind.IMAGE:
            logger.info("Step2 — vidéo ignorée en P0 pour %s", url)
            continue
        downloaded_images.append(media)

    if not downloaded_images:
        logger.info("Step2 — aucun média image exploitable, fallback texte seul.")
        return build_llm_messages(prompt_text, [])

    return build_message_with_media(prompt_text, downloaded_images)


def _default_media_fetcher() -> MediaFetcher:
    settings = get_settings()
    client = httpx.Client(timeout=max(settings.media_image_timeout, settings.media_video_timeout))
    return MediaFetcher(
        client=client,
        max_size_bytes=settings.media_max_size_bytes,
        image_timeout=settings.media_image_timeout,
        video_timeout=settings.media_video_timeout,
        max_retries=settings.media_max_retries,
    )


def _convert_and_filter(
    llm_output: DetectedProductList,
    ad_id: str,
) -> list[DetectedProduct]:
    """Convertit les sorties LLM en DetectedProduct et filtre par importance.

    La conversion ``color`` (string → Color enum) est robuste : si la valeur
    LLM ne correspond à aucun enum, on tente un matching insensible à la casse,
    et en dernier recours on utilise ``Color.MULTICOLOR`` avec un warning.

    Args:
        llm_output: La liste brute produite par le LLM.
        ad_id: ID de la pub (pour les logs).

    Returns:
        Liste de ``DetectedProduct`` filtrés et triés.
    """
    products: list[DetectedProduct] = []

    for raw in llm_output.products:
        if raw.importance <= IMPORTANCE_THRESHOLD:
            logger.debug(
                "Step2 — Produit filtré (importance=%d) : %s",
                raw.importance,
                raw.raw_description[:60],
            )
            continue

        # Conversion robuste de la couleur
        color = _normalize_color(raw.color, ad_id)

        products.append(
            DetectedProduct(
                raw_description=raw.raw_description,
                universe=raw.universe,
                color=color,
                importance=raw.importance,
            )
        )

    # Trier par importance décroissante : les produits principaux d'abord
    return sorted(products, key=lambda p: p.importance, reverse=True)


def _normalize_color(raw_color: str, ad_id: str) -> Color:
    """Normalise une couleur string LLM vers l'enum Color.

    Stratégie :
        1. Match exact (ex: "Black" → Color.BLACK).
        2. Match insensible à la casse.
        3. Fallback → Color.MULTICOLOR avec warning.

    Args:
        raw_color: Valeur couleur brute du LLM.
        ad_id: ID de la pub pour le logging.

    Returns:
        Instance ``Color`` normalisée.
    """
    # 1. Match exact
    try:
        return Color(raw_color)
    except ValueError:
        pass

    # 2. Match insensible à la casse
    raw_lower = raw_color.strip().lower()
    for color in Color:
        if color.value.lower() == raw_lower:
            return color

    # 3. Fallback
    logger.warning(
        "Step2 — Couleur LLM non reconnue '%s' pour ad_id=%s → Multicolor.",
        raw_color,
        ad_id,
    )
    return Color.MULTICOLOR
