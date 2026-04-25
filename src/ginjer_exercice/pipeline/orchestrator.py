"""Orchestrateur minimal du pipeline produit."""

from __future__ import annotations

import logging

import httpx

from ..config import get_settings
from ..data_access.catalog_provider import get_catalog_provider
from ..data_access.media_fetcher import MediaFetcher
from ..data_access.results_repository import ResultsRepository
from ..llm.base import LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.runtime_warnings import collect_runtime_warnings
from ..observability.scoring import score_confidence, score_taxonomy_coherence
from ..observability.tracing import pipeline_trace
from ..schemas.ad import Ad
from ..schemas.pipeline import PipelineOutput
from ..schemas.products import FinalProductLabel
from ..schemas.scores import ScoreReport
from ..taxonomy.loader import load_taxonomy
from ..web_search.null_provider import NullWebSearchProvider
from . import step1_universe, step2_products, step3_classify, step4_name
from .step5_fallback import step5_fallback

logger = logging.getLogger(__name__)


def run_ad(
    ad: Ad,
    *,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    results_repository: ResultsRepository,
) -> PipelineOutput:
    """Execute les steps 1 a 4 pour une pub, puis persiste le resultat."""
    logger.info("orchestrator.run_ad started for ad_id=%s brand=%s", ad.platform_ad_id, ad.brand.value)
    taxonomy = load_taxonomy(ad.brand)
    settings = get_settings()
    catalog_provider = get_catalog_provider(ad.brand)
    web_search_provider = NullWebSearchProvider()
    ad_context = ad.all_text()

    with collect_runtime_warnings() as runtime_warnings:
        with pipeline_trace(ad, session_id="process-ad") as trace:
            universe_result = step1_universe.execute(
                ad,
                llm_provider=llm_provider,
                prompt_registry=prompt_registry,
                trace=trace,
            )
            with httpx.Client(timeout=max(settings.media_image_timeout, settings.media_video_timeout)) as http_client:
                media_fetcher = MediaFetcher(
                    client=http_client,
                    max_size_bytes=settings.media_max_size_bytes,
                    image_timeout=settings.media_image_timeout,
                    video_timeout=settings.media_video_timeout,
                    max_retries=settings.media_max_retries,
                )
                products = step2_products.execute(
                    ad,
                    universe_result,
                    llm_provider=llm_provider,
                    prompt_registry=prompt_registry,
                    trace=trace,
                    media_fetcher=media_fetcher,
                )

            final_products: list[FinalProductLabel] = []
            for product in products:
                logger.debug(
                    "Processing detected product for ad_id=%s description=%s importance=%d",
                    ad.platform_ad_id,
                    product.raw_description[:80],
                    product.importance,
                )
                classification = step3_classify.execute(
                    product,
                    ad,
                    taxonomy,
                    llm_provider=llm_provider,
                    prompt_registry=prompt_registry,
                    trace=trace,
                )
                name_info = step4_name.execute(
                    product,
                    classification,
                    ad,
                    llm_provider=llm_provider,
                    prompt_registry=prompt_registry,
                    trace=trace,
                )
                if name_info is None:
                    name_info = step5_fallback(
                        product=product,
                        classification=classification,
                        brand=ad.brand,
                        ad_context=ad_context,
                        llm_provider=llm_provider,
                        prompt_registry=prompt_registry,
                        trace_context=trace,
                        catalog_provider=catalog_provider,
                        web_search_provider=web_search_provider,
                        confidence_threshold=settings.fallback_confidence_threshold,
                    )
                final_products.append(
                    FinalProductLabel(
                        detected=product,
                        classification=classification,
                        name_info=name_info,
                    )
                )

            trace_id = trace.trace_id or f"local-{ad.platform_ad_id}"
            output = PipelineOutput(
                ad_id=ad.platform_ad_id,
                brand=ad.brand,
                products=final_products,
                warnings=list(runtime_warnings),
                scores=ScoreReport(),
                trace_id=trace_id,
            )
            output.scores = ScoreReport(
                taxonomy_coherence=score_taxonomy_coherence(ad.brand, output, trace_id=trace_id),
                confidence=score_confidence(output, trace_id=trace_id),
                llm_judge=None,
            )
            trace.update_output(output.model_dump())
            if output.warnings:
                trace.update_metadata(runtime_warnings=output.warnings, warning_count=len(output.warnings))
            results_repository.save(output)

            if output.warnings:
                logger.warning(
                    "orchestrator.run_ad completed with %d warning(s) for ad_id=%s",
                    len(output.warnings),
                    output.ad_id,
                )
            logger.info(
                "orchestrator.run_ad completed for ad_id=%s products=%d needs_review=%s taxonomy_coherence=%.3f confidence=%.3f",
                output.ad_id,
                len(output.products),
                output.needs_review,
                output.scores.taxonomy_coherence or 0.0,
                output.scores.confidence or 0.0,
            )
            return output
