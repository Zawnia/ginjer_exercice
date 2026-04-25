"""Step 5a fallback using enriched LLM prompting and optional web verification."""

from __future__ import annotations

import json
import logging

from ..data_access.catalog_provider import CatalogProvider
from ..llm.base import LLMCallConfig, LLMMessage, LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.runtime_warnings import add_runtime_warning
from ..observability.tracing import TraceContext
from ..schemas.ad import Brand
from ..schemas.products import DetectedProduct, ProductClassification, ProductName
from ..schemas.step_outputs import FallbackNameSuggestion
from ..web_search.base import WebSearchProvider

logger = logging.getLogger(__name__)

_PROMPT_NAME = "pipeline/fallback_enriched"
_LOW_CONFIDENCE_WARNING = "step5a: low confidence name, flagged for review"
_NO_NAME_WARNING = "step5a: could not identify product name"


def step5_fallback(
    product: DetectedProduct,
    classification: ProductClassification,
    brand: Brand,
    ad_context: str,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    trace_context: TraceContext,
    catalog_provider: CatalogProvider | None = None,
    web_search_provider: WebSearchProvider | None = None,
    confidence_threshold: float = 0.7,
) -> ProductName:
    """Suggest a product name via enriched LLM fallback and optional web verification.

    Args:
        product: Product detected by Step 2.
        classification: Product classification returned by Step 3.
        brand: Brand currently being processed.
        ad_context: Concatenated advertisement text context.
        llm_provider: Structured-output LLM provider.
        prompt_registry: Prompt registry used to resolve the fallback prompt.
        trace_context: Parent trace context for observability.
        catalog_provider: Optional provider returning a reference subset.
        web_search_provider: Optional provider used for future Step 5b verification.
        confidence_threshold: Minimum confidence required to accept the name.

    Returns:
        Final fallback product naming result.
    """
    catalog_subset = []
    if catalog_provider is not None:
        catalog_subset = catalog_provider.get_subset(
            brand=brand,
            universe=classification.universe,
            category=classification.category,
            limit=50,
        )

    prompt = prompt_registry.get(_PROMPT_NAME)
    compiled = prompt.compile(
        brand=brand.value,
        universe=classification.universe,
        category=classification.category,
        subcategory=classification.subcategory,
        product_type=classification.product_type or "",
        color=product.color.value,
        importance=str(product.importance),
        visual_description=product.raw_description,
        ad_context=ad_context or "(no text available)",
        catalog_subset=_format_catalog_subset(catalog_subset),
    )

    messages = [LLMMessage.from_text(compiled)]
    llm_config = LLMCallConfig(
        model_name=prompt.config.get("model", "gemini-2.5-flash"),
        temperature=prompt.config.get("temperature", 0.0),
        max_tokens=prompt.config.get("max_tokens", 4000),
    )

    with trace_context.child_span(
        name="step_5a_fallback_llm",
        input_payload={
            "brand": brand.value,
            "product": product.model_dump(),
            "classification": classification.model_dump(),
            "catalog_subset_size": len(catalog_subset),
        },
    ) as span:
        response = llm_provider.generate_structured(
            messages=messages,
            response_model=FallbackNameSuggestion,
            config=llm_config,
            trace_context=span,
        )
        suggestion: FallbackNameSuggestion = response.parsed  # type: ignore[assignment]
        span.update_output(suggestion.model_dump())

    if suggestion.name is None:
        add_runtime_warning(_NO_NAME_WARNING)
        return ProductName(
            name=None,
            source="fallback_failed",
            confidence=0.0,
            needs_review=True,
            sources_consulted=_sources_consulted(catalog_subset, False),
        )

    if web_search_provider is not None and suggestion.confidence < confidence_threshold:
        with trace_context.child_span(
            name="step_5b_web_verify",
            input_payload={
                "brand": brand.value,
                "suggested_name": suggestion.name,
                "classification": classification.model_dump(),
            },
        ) as span:
            verification = web_search_provider.verify_product_name(brand, suggestion.name, classification)
            if verification.confirmed and verification.verified_name and verification.confidence >= confidence_threshold:
                result = ProductName(
                    name=verification.verified_name,
                    source="fallback_web",
                    confidence=verification.confidence,
                    needs_review=False,
                    sources_consulted=_sources_consulted(catalog_subset, True),
                )
                span.update_output({"skipped": False, **verification.model_dump(), "result": result.model_dump()})
                return result

            reason = "unconfirmed"
            if web_search_provider.__class__.__name__ == "NullWebSearchProvider":
                reason = "null_provider"
            span.update_output({"skipped": True, "reason": reason, **verification.model_dump()})

    if suggestion.confidence >= confidence_threshold:
        return ProductName(
            name=suggestion.name,
            source="fallback_enriched",
            confidence=suggestion.confidence,
            needs_review=False,
            sources_consulted=_sources_consulted(catalog_subset, False),
        )

    add_runtime_warning(_LOW_CONFIDENCE_WARNING)
    return ProductName(
        name=suggestion.name,
        source="fallback_enriched",
        confidence=suggestion.confidence,
        needs_review=True,
        sources_consulted=_sources_consulted(catalog_subset, web_search_provider is not None),
    )


def _format_catalog_subset(catalog_subset: list[dict[str, object]]) -> str:
    """Serialize the catalog subset for prompt injection.

    Args:
        catalog_subset: Reference entries selected for the current product.

    Returns:
        A compact JSON string or ``[]`` when no subset is available.
    """
    if not catalog_subset:
        return "[]"
    return json.dumps(catalog_subset, ensure_ascii=False, indent=2)


def _sources_consulted(catalog_subset: list[dict[str, object]], used_web_search: bool) -> list[str]:
    """Build the list of sources consulted by Step 5.

    Args:
        catalog_subset: Catalog subset used during prompting.
        used_web_search: Whether Step 5b verification was attempted.

    Returns:
        A list of source labels attached to ``ProductName``.
    """
    sources = ["fallback_enriched_llm"]
    if catalog_subset:
        sources.append("catalog_reference")
    if used_web_search:
        sources.append("web_search_stub")
    return sources
