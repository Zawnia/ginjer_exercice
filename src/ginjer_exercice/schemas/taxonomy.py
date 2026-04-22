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

        N'accepte pas la sentinelle comme sous-catégorie valide : appeler
        ``is_terminal_category()`` pour les catégories terminales.

        Args:
            universe: Nom de l'univers.
            category: Nom de la catégorie.
            subcategory: Nom de la sous-catégorie à vérifier.

        Returns:
            True si la combinaison existe et n'est pas une valeur sentinelle.
        """
        if subcategory == NO_SUBCATEGORY_SENTINEL:
            return False
        try:
            return subcategory in self.tree[universe][category]
        except KeyError:
            return False

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
