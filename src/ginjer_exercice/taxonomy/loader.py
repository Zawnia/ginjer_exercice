from __future__ import annotations
import logging
from pathlib import Path

from ..exceptions import TaxonomyNotFoundError
from ..schemas.ad import Brand
from ..schemas.taxonomy import BrandTaxonomy
from .product_categorisation_parser import (
    parse_canonical_taxonomy,
    _DEFAULT_CANONICAL_SOURCE,
)
from .store import TaxonomyStore

logger = logging.getLogger(__name__)


def load_taxonomy(
    brand: Brand,
    force_refresh: bool = False,
    store: TaxonomyStore | None = None,
    canonical_source_path: Path | None = None,
) -> BrandTaxonomy:
    """Charge la taxonomie applicable à une marque.

    Stratégie de résolution (dans l'ordre) :
        1. Taxonomie spécialisée par marque (``{brand.value}.json``), si elle existe.
           Phase actuelle : ce fichier n'est jamais créé automatiquement.
           Extension future : enrichissements marque-spécifiques persistés ici.
        2. Taxonomie canonique commune déjà persistée (``canonical.json``).
        3. Bootstrap : reparse le fichier source et persiste la canonique.

    Quand ``force_refresh`` est True, les étapes 1 et 2 sont ignorées.

    Args:
        brand: Marque pour laquelle la taxonomie est demandée.
        force_refresh: Si True, ignore les caches et reparse la source.
        store: Backend de persistence (injecté pour les tests).
        canonical_source_path: Chemin vers ``product_categorisation.json``.
            Défaut : chemin résolu depuis la racine du projet.

    Returns:
        Une instance ``BrandTaxonomy`` validée.

    Raises:
        FileNotFoundError: Si aucune taxonomie n'est disponible et que le
            fichier source est introuvable.
        ValueError: Si la structure du fichier source est invalide.
    """
    store = store or TaxonomyStore()
    source_path = canonical_source_path or _DEFAULT_CANONICAL_SOURCE

    if not force_refresh:
        try:
            return store.load_taxonomy(brand.value)
        except TaxonomyNotFoundError:
            pass

        try:
            return store.load_taxonomy("canonical")
        except TaxonomyNotFoundError:
            pass

    canonical_taxonomy = parse_canonical_taxonomy(source_path)
    store.save_taxonomy("canonical", canonical_taxonomy, source=str(source_path))

    return canonical_taxonomy
