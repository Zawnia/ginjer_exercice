"""Step 3 — Taxonomy classification (avant step 2, conformément au plan).

Classifie un ``DetectedProduct`` dans l'arbre taxonomique de la marque.
C'est l'étape avec la logique métier la plus complexe : validation taxonomique
et mécanisme de retry avec message correctif si le LLM renvoie un chemin invalide.

Usage::

    from ginjer_exercice.pipeline import step3_classify
    from ginjer_exercice.taxonomy.loader import load_taxonomy

    taxonomy = load_taxonomy(brand)
    classification = step3_classify.execute(
        product,
        ad,
        taxonomy,
        llm_provider=llm,
        prompt_registry=registry,
        trace=trace,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from ..exceptions import LLMValidationError
from ..llm.base import LLMCallConfig, LLMMessage, LLMProvider
from ..observability.prompts import PromptRegistry
from ..observability.tracing import step_span
from ..schemas.ad import Ad
from ..schemas.products import DetectedProduct, ProductClassification
from ..schemas.taxonomy import BrandTaxonomy
from ._helpers import build_llm_messages, build_texts_block

logger = logging.getLogger(__name__)

_PROMPT_NAME = "pipeline/classification"
MAX_VALIDATION_RETRIES = 2


def execute(
    product: DetectedProduct,
    ad: Ad,
    taxonomy: BrandTaxonomy,
    *,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    trace: Any,
) -> ProductClassification:
    """Classifie un produit détecté dans la taxonomie de la marque.

    Injecte un slice filtré de la taxonomie (uniquement l'univers du produit)
    pour réduire le contexte LLM. Si le LLM renvoie une combinaison invalide
    (chemin inexistant dans la taxonomie), injecte un message correctif et
    retente jusqu'à ``MAX_VALIDATION_RETRIES`` fois.

    Chaque tentative génère une génération LLM enfant distincte dans Langfuse,
    permettant de tracer la séquence de corrections.

    Args:
        product: Le produit à classifier (sortie de step2, incluant universe).
        ad: La publicité source (pour le contexte de marque).
        taxonomy: La taxonomie de la marque (filtrée par univers en interne).
        llm_provider: Fournisseur LLM.
        prompt_registry: Registre des prompts versionnés.
        trace: Contexte de trace Langfuse (span parent).

    Returns:
        ``ProductClassification`` avec un chemin taxonomique valide et une
        confiance entre 0 et 1.

    Raises:
        LLMValidationError: Si après ``MAX_VALIDATION_RETRIES`` tentatives,
            le LLM n'a pas pu produire un chemin taxonomique valide.
    """
    with step_span(
        name="step_3_classify",
        input_payload={
            "product": product.model_dump(),
            "ad_id": ad.platform_ad_id,
            "brand": ad.brand.value,
        },
    ):
        # 1. Récupérer le prompt
        prompt = prompt_registry.get(_PROMPT_NAME)

        # 2. Préparer la taxonomie filtrée par univers du produit
        taxo_slice = taxonomy.slice_for_universe(product.universe)
        taxonomy_tree = taxo_slice.format_as_bullet_list()

        # 3. Compiler le prompt
        compiled = prompt.compile(
            brand=ad.brand.value,
            product_description=product.raw_description,
            product_color=product.color.value,
            product_universe=product.universe,
            taxonomy_tree=taxonomy_tree,
        )

        # 4. Configurer l'appel LLM
        llm_config = LLMCallConfig(
            model_name=prompt.config.get("model", "gemini-2.0-flash"),
            temperature=prompt.config.get("temperature", 0.1),
            max_tokens=prompt.config.get("max_tokens", 1500),
        )

        # 5. Construire les messages initiaux (texte + médias de la pub)
        messages = build_llm_messages(compiled, ad.media_urls)

        # 6. Boucle de retry avec validation métier
        last_parsed: ProductClassification | None = None

        for attempt in range(MAX_VALIDATION_RETRIES + 1):
            logger.debug(
                "Step3 — Tentative %d/%d pour ad_id=%s, product=%s",
                attempt + 1,
                MAX_VALIDATION_RETRIES + 1,
                ad.platform_ad_id,
                product.raw_description[:50],
            )

            response = llm_provider.generate_structured(
                messages=messages,
                response_model=ProductClassification,
                config=llm_config,
                trace_context=None,
            )
            parsed: ProductClassification = response.parsed  # type: ignore[assignment]
            last_parsed = parsed

            # Validation taxonomique (insensible à la casse grâce à is_valid_path)
            is_terminal = taxonomy.is_terminal_category(parsed.universe, parsed.category)
            is_valid = is_terminal or taxonomy.is_valid_path(
                parsed.universe, parsed.category, parsed.subcategory
            )

            if is_valid:
                logger.info(
                    "Step3 — Classifié en %d tentative(s) : %s > %s > %s (conf=%.2f)",
                    attempt + 1,
                    parsed.universe,
                    parsed.category,
                    parsed.subcategory,
                    parsed.confidence,
                )
                return parsed

            # Chemin invalide : construire le message correctif
            valid_subcats = taxonomy.list_valid_subcategories(parsed.universe, parsed.category)
            valid_subcats_str = (
                ", ".join(f'"{s}"' for s in valid_subcats)
                if valid_subcats
                else "(no valid subcategories — this may be a terminal category)"
            )

            correction_msg = (
                f"The combination you provided does not exist in the taxonomy.\n"
                f"Universe: '{parsed.universe}', Category: '{parsed.category}', "
                f"Subcategory: '{parsed.subcategory}' — this path is INVALID.\n"
                f"Valid subcategories for '{parsed.category}' under '{parsed.universe}' are: "
                f"{valid_subcats_str}.\n"
                f"Please provide a corrected classification using ONLY values from the taxonomy tree."
            )

            logger.warning(
                "Step3 — Chemin taxonomique invalide (tentative %d) : %s > %s > %s",
                attempt + 1,
                parsed.universe,
                parsed.category,
                parsed.subcategory,
            )

            # Ajouter la réponse du LLM et le message correctif à la conversation
            messages.append(LLMMessage(text=parsed.model_dump_json(), media=[]))
            messages.append(LLMMessage(text=correction_msg, media=[]))

        # Toutes les tentatives ont échoué
        raise LLMValidationError(
            f"Step3 — Impossible d'obtenir un chemin taxonomique valide après "
            f"{MAX_VALIDATION_RETRIES + 1} tentatives pour le produit : "
            f"'{product.raw_description[:100]}'. "
            f"Dernière classification : {last_parsed.model_dump() if last_parsed else 'N/A'}"
        )
