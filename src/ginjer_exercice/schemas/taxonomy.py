from __future__ import annotations
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Valeur sentinelle pour les catégories terminales sans sous-catégorie explicite.
# Convention interne uniquement — ne doit jamais fuiter dans les prompts LLM
# ni dans les outputs finaux du pipeline.
NO_SUBCATEGORY_SENTINEL = "__NO_SUBCATEGORY__"


class BrandTaxonomy(BaseModel):
    """Structure arborescente de la taxonomie : universe -> category -> subcategory[].

    Les catégories sans sous-catégorie (ex : Art & Culture, Pets) sont représentées
    par la valeur sentinelle ``NO_SUBCATEGORY_SENTINEL``. Utiliser ``is_terminal_category()``
    pour les détecter proprement.

    Architecture :
        - Phase actuelle : une taxonomie canonique commune à toutes les marques,
          extraite de ``product_categorisation.json``.
        - Extension future : taxonomies spécialisées par marque (chanel.json, etc.)
          chargées en priorité par ``load_taxonomy``.
    """

    tree: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description="Arbre taxonomique : universe -> category -> [subcategories]",
    )

    def get_universes(self) -> list[str]:
        """Retourne la liste triée de tous les univers.

        Returns:
            Liste des noms d'univers présents dans la taxonomie.
        """
        return sorted(self.tree.keys())

    def get_categories(self, universe: str) -> list[str]:
        """Retourne la liste triée des catégories pour un univers donné.

        Args:
            universe: Nom de l'univers (ex: ``"Women"``, ``"Beauty"``).

        Returns:
            Liste des catégories ou liste vide si l'univers est inconnu.
        """
        if universe not in self.tree:
            return []
        return sorted(self.tree[universe].keys())

    def get_subcategories(self, universe: str, category: str) -> list[str]:
        """Retourne les sous-catégories pour un couple (universe, category).

        Args:
            universe: Nom de l'univers.
            category: Nom de la catégorie.

        Returns:
            Liste de sous-catégories (peut contenir ``NO_SUBCATEGORY_SENTINEL``
            pour les catégories terminales). Liste vide si introuvable.
        """
        try:
            return self.tree[universe][category]
        except KeyError:
            return []

    def is_terminal_category(self, universe: str, category: str) -> bool:
        """Indique si une catégorie est terminale (sans vraie sous-catégorie).

        Une catégorie est considérée terminale quand ses seules sous-catégories
        sont la valeur sentinelle ``NO_SUBCATEGORY_SENTINEL``.

        Args:
            universe: Nom de l'univers.
            category: Nom de la catégorie.

        Returns:
            True si la catégorie n'a pas de sous-catégorie réelle.
        """
        subcats = self.get_subcategories(universe, category)
        return subcats == [NO_SUBCATEGORY_SENTINEL] or subcats == []

    def is_valid_path(self, universe: str, category: str, subcategory: str) -> bool:
        """Vérifie si le chemin taxonomique (universe, category, subcategory) est valide.

        La comparaison est insensible à la casse et aux espaces superflus, car le LLM
        peut renvoyer ``"Handbags"`` alors que la taxonomie stocke ``"handbags"``.
        N'accepte pas la sentinelle comme sous-catégorie valide.

        Args:
            universe: Nom de l'univers.
            category: Nom de la catégorie.
            subcategory: Nom de la sous-catégorie à vérifier.

        Returns:
            True si la combinaison existe (matching insensible à la casse) et
            n'est pas une valeur sentinelle.
        """
        if subcategory == NO_SUBCATEGORY_SENTINEL:
            return False

        def _norm(s: str) -> str:
            return s.strip().lower()

        norm_universe = _norm(universe)
        norm_category = _norm(category)
        norm_subcategory = _norm(subcategory)

        for u_key, cats in self.tree.items():
            if _norm(u_key) != norm_universe:
                continue
            for c_key, subcats in cats.items():
                if _norm(c_key) != norm_category:
                    continue
                return any(_norm(s) == norm_subcategory for s in subcats)
        return False

    def slice_for_universe(self, universe: str) -> "BrandTaxonomy":
        """Retourne une taxonomie restreinte à un univers spécifique.

        Utilisé par step3 pour injecter uniquement les branches pertinentes
        dans le prompt de classification, évitant de surcharger le contexte LLM.

        La recherche est insensible à la casse.

        Args:
            universe: Nom de l'univers à extraire.

        Returns:
            Un ``BrandTaxonomy`` ne contenant que l'univers demandé.
            Retourne une taxonomie vide si l'univers est inconnu.
        """
        norm = universe.strip().lower()
        matched = {
            k: v for k, v in self.tree.items() if k.strip().lower() == norm
        }
        return BrandTaxonomy(tree=matched)

    def list_valid_subcategories(self, universe: str, category: str) -> list[str]:
        """Retourne les sous-catégories valides pour un couple (universe, category).

        Exclut la valeur sentinelle. Utilisé dans les messages de retry de step3
        pour corriger le LLM avec des valeurs concrètes.

        La recherche est insensible à la casse.

        Args:
            universe: Nom de l'univers.
            category: Nom de la catégorie.

        Returns:
            Liste des sous-catégories réelles (sans sentinelle).
        """
        norm_u = universe.strip().lower()
        norm_c = category.strip().lower()

        for u_key, cats in self.tree.items():
            if u_key.strip().lower() != norm_u:
                continue
            for c_key, subcats in cats.items():
                if c_key.strip().lower() != norm_c:
                    continue
                return [s for s in subcats if s != NO_SUBCATEGORY_SENTINEL]
        return []

    def format_as_bullet_list(self, universe_filter: str | None = None) -> str:
        """Formate la taxonomie en liste à puces hiérarchique lisible par un LLM.

        Le LLM comprend mieux une liste à puces qu'un objet JSON imbriqué.
        Exemple de sortie ::

            - Bags
              - Handbags: Top handle, Hobo, Bucket
              - Crossbody/Shoulder Bag: Crossbody, Shoulder
            - Shoes
              - Heels: Pumps, Slingbacks

        Args:
            universe_filter: Si fourni, restreint la sortie à cet univers.

        Returns:
            Chaîne multi-lignes de la taxonomie formatée.
        """
        lines: list[str] = []
        tree = self.tree

        if universe_filter:
            norm = universe_filter.strip().lower()
            tree = {k: v for k, v in tree.items() if k.strip().lower() == norm}

        for universe, cats in sorted(tree.items()):
            lines.append(f"**{universe}**")
            for category, subcats in sorted(cats.items()):
                real_subcats = [s for s in subcats if s != NO_SUBCATEGORY_SENTINEL]
                if real_subcats:
                    lines.append(f"  - {category}: {', '.join(sorted(real_subcats))}")
                else:
                    lines.append(f"  - {category}")

        return "\n".join(lines)

    def serialize(self) -> str:
        """Sérialise la taxonomie en chaîne JSON.

        Returns:
            Représentation JSON indentée de la taxonomie.
        """
        return self.model_dump_json(indent=2)

    @classmethod
    def deserialize(cls, json_data: str | dict) -> BrandTaxonomy:
        """Désérialise une taxonomie depuis une chaîne JSON ou un dictionnaire.

        Args:
            json_data: Chaîne JSON ou dictionnaire Python.

        Returns:
            Instance ``BrandTaxonomy`` validée.
        """
        if isinstance(json_data, str):
            return cls.model_validate_json(json_data)
        return cls.model_validate(json_data)
