from __future__ import annotations
import json
import logging
from pathlib import Path

from src.ginjer_exercice.schemas.taxonomy import BrandTaxonomy, NO_SUBCATEGORY_SENTINEL

logger = logging.getLogger(__name__)

# Chemin résolu depuis la racine du projet (parents[3] depuis src/ginjer_exercice/taxonomy/)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CANONICAL_SOURCE = _PROJECT_ROOT / "data" / "raw" / "product_categorisation.json"


def _extract_anyof_branches(data: dict) -> list[dict]:
    """Extrait et valide la liste des branches ``anyOf`` du schéma produit.

    Args:
        data: Contenu parsé du fichier ``product_categorisation.json``.

    Returns:
        La liste des objets de branches ``anyOf``.

    Raises:
        ValueError: Si le chemin vers ``anyOf`` est introuvable ou de type inattendu.
    """
    try:
        any_of = (
            data["schema"]
            ["properties"]["products"]
            ["items"]["properties"]
            ["product_categorisation"]
            ["anyOf"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Structure inattendue dans product_categorisation.json : "
            "chemin schema.properties.products.items.properties"
            ".product_categorisation.anyOf introuvable."
        ) from exc

    if not isinstance(any_of, list):
        raise ValueError(
            f"'anyOf' doit être une liste, reçu : {type(any_of).__name__}"
        )

    return any_of


def _safe_string_enum(properties: dict, field: str) -> list[str]:
    """Extrait et valide une liste d'enum string depuis un champ de propriété.

    Les valeurs non-string sont silencieusement filtrées avec un avertissement.

    Args:
        properties: Dictionnaire des propriétés d'une branche ``anyOf``.
        field: Nom du champ (ex: ``"universe"``, ``"subcategory"``).

    Returns:
        Liste de chaînes de caractères (peut être vide).
    """
    raw: list = properties.get(field, {}).get("enum", [])
    if not isinstance(raw, list):
        logger.warning("Le champ '%s' a un enum non-liste (%s) — branche ignorée.", field, type(raw).__name__)
        return []
    valid = [x for x in raw if isinstance(x, str)]
    if len(valid) != len(raw):
        logger.warning(
            "Le champ '%s' contenait %d valeur(s) non-string filtrée(s).",
            field, len(raw) - len(valid),
        )
    return valid


def parse_canonical_taxonomy(json_path: str | Path) -> BrandTaxonomy:
    """Extrait la taxonomie canonique depuis le JSON Schema de référence.

    Parcourt les branches ``anyOf`` de ``product_categorisation.json`` pour
    construire un arbre ``universe -> category -> [subcategory]``. Les catégories
    terminales (sans sous-catégorie déclarée, ex: Art & Culture) sont représentées
    par la valeur sentinelle ``NO_SUBCATEGORY_SENTINEL``.

    Les branches mal formées (non-dict) sont comptabilisées et signalées en warning
    plutôt que silencieusement ignorées.

    Args:
        json_path: Chemin vers le fichier ``product_categorisation.json``.

    Returns:
        Une instance ``BrandTaxonomy`` avec l'arbre canonique complet.

    Raises:
        FileNotFoundError: Si le fichier est introuvable au chemin donné.
        ValueError: Si la structure du fichier est invalide ou incohérente.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"Fichier de taxonomie de référence introuvable : {json_path.resolve()}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    branches = _extract_anyof_branches(data)

    tree: dict[str, dict[str, set[str]]] = {}
    skipped = 0

    for branch in branches:
        if not isinstance(branch, dict):
            skipped += 1
            continue

        properties = branch.get("properties", {})

        universes = _safe_string_enum(properties, "universe")
        categories = _safe_string_enum(properties, "category")
        raw_subcats = _safe_string_enum(properties, "subcategory")
        subcategories = raw_subcats if raw_subcats else [NO_SUBCATEGORY_SENTINEL]

        for universe in universes:
            tree.setdefault(universe, {})
            for category in categories:
                tree[universe].setdefault(category, set())
                for subcategory in subcategories:
                    tree[universe][category].add(subcategory)

    if skipped:
        logger.warning(
            "%d branche(s) anyOf ignorée(s) car non-dict (structure inattendue).", skipped
        )

    final_tree: dict[str, dict[str, list[str]]] = {
        u: {c: sorted(subcats) for c, subcats in cats.items()}
        for u, cats in tree.items()
    }

    return BrandTaxonomy(tree=final_tree)
