"""Pipeline package — étapes métier du pipeline de détection de produits.

Chaque step est une fonction ``execute(...)`` indépendante :
    - Les inputs métier sont en arguments positionnels (objets Pydantic).
    - Les dépendances techniques sont en keyword-only (après le ``*,``).
    - Chaque step crée son propre span Langfuse enfant via ``step_span()``.
    - Une step ne connaît pas ses voisines.

Ordre d'exécution dans l'orchestrator (Phase 6) :
    Step 1 → Step 2 → (pour chaque produit) Step 3 → Step 4 → [Step 5 si besoin]
"""

from . import step1_universe, step2_products, step3_classify, step4_name

__all__ = [
    "step1_universe",
    "step2_products",
    "step3_classify",
    "step4_name",
]
