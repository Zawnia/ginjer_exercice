"""Orchestrateur minimal du pipeline produit."""

from __future__ import annotations

import logging

from ..data_access.results_repository import ResultsRepository
from ..llm.base import LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.scoring import score_confidence, score_taxonomy_coherence
from ..observability.tracing import pipeline_trace
from ..schemas.ad import Ad
from ..schemas.pipeline import PipelineOutput
from ..schemas.products import FinalProductLabel
from ..schemas.scores import ScoreReport
from ..taxonomy.loader import load_taxonomy
from . import step1_universe, step2_products, step3_classify, step4_name

logger = logging.getLogger(__name__)


def run_ad(
    ad: Ad,
    *,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    results_repository: ResultsRepository,
) -> PipelineOutput:
    """Exécute les steps 1 à 4 pour une pub, puis persiste le résultat."""
    logger.info("orchestrator.run_ad started for ad_id=%s brand=%s", ad.platform_ad_id, ad.brand.value)
    taxonomy = load_taxonomy(ad.brand)

    with pipeline_trace(ad, session_id="process-ad") as trace:
        universe_result = step1_universe.execute(
            ad,
            llm_provider=llm_provider,
            prompt_registry=prompt_registry,
            trace=trace,
        )
        products = step2_products.execute(
            ad,
            universe_result,
            llm_provider=llm_provider,
            prompt_registry=prompt_registry,
            trace=trace,
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
            scores=ScoreReport(),
            trace_id=trace_id,
        )
        output.scores = ScoreReport(
            taxonomy_coherence=score_taxonomy_coherence(ad.brand, output, trace_id=trace_id),
            confidence=score_confidence(output, trace_id=trace_id),
            llm_judge=None,
        )
        trace.update_output(output.model_dump())
        results_repository.save(output)
        logger.info(
            "orchestrator.run_ad completed for ad_id=%s products=%d needs_review=%s taxonomy_coherence=%.3f confidence=%.3f",
            output.ad_id,
            len(output.products),
            output.needs_review,
            output.scores.taxonomy_coherence or 0.0,
            output.scores.confidence or 0.0,
        )
        return output
