import logging

from ..schemas.ad import Brand
from ..schemas.pipeline import PipelineOutput
from ..taxonomy.loader import load_taxonomy
from .client import get_langfuse_client

logger = logging.getLogger(__name__)


def score_taxonomy_coherence(
    brand: Brand,
    products: PipelineOutput | list,
    trace_id: str | None = None,
    observation_id: str | None = None,
) -> float:
    """Calcule la cohérence taxonomique moyenne d'une sortie pipeline."""
    final_products = products.products if isinstance(products, PipelineOutput) else products
    if not final_products:
        score_value = 0.0
    else:
        taxonomy = load_taxonomy(brand)
        validations = []
        for product in final_products:
            classification = product.classification
            is_terminal = taxonomy.is_terminal_category(
                classification.universe,
                classification.category,
            )
            is_valid = is_terminal or taxonomy.is_valid_path(
                classification.universe,
                classification.category,
                classification.subcategory,
            )
            validations.append(1.0 if is_valid else 0.0)
        score_value = sum(validations) / len(validations)

    _publish_score(
        trace_id=trace_id,
        observation_id=observation_id,
        name="taxonomy_coherence",
        value=score_value,
        comment=f"[{brand.value}] taxonomic path validity",
    )
    return score_value


def score_confidence(
    products: PipelineOutput | list,
    trace_id: str | None = None,
    observation_id: str | None = None,
) -> float:
    """Calcule la confiance moyenne par produit à partir des sorties disponibles."""
    final_products = products.products if isinstance(products, PipelineOutput) else products
    if not final_products:
        score_value = 0.0
    else:
        per_product_scores = []
        for product in final_products:
            values = [product.classification.confidence]
            if product.name_info is not None:
                values.append(product.name_info.confidence)
            per_product_scores.append(sum(values) / len(values))
        score_value = sum(per_product_scores) / len(per_product_scores)

    _publish_score(
        trace_id=trace_id,
        observation_id=observation_id,
        name="confidence",
        value=score_value,
    )
    return score_value


def score_llm_judge(
    trace_id: str | None = None,
    observation_id: str | None = None,
    **kwargs,
) -> float:
    raise NotImplementedError("LLM-as-judge sera implémenté en phase 7")


def _publish_score(
    *,
    trace_id: str | None,
    observation_id: str | None,
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    langfuse = get_langfuse_client()
    if langfuse is None or trace_id is None:
        return

    try:
        langfuse.create_score(
            trace_id=trace_id,
            observation_id=observation_id,
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as exc:
        logger.warning("Impossible de logger le score '%s' dans Langfuse: %s", name, exc)
