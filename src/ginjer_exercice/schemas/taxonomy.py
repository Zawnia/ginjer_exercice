from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any

class TaxonomyNode(BaseModel):
    '''Représentation générique d'un nœud dans la taxonomie, pourrait être utile si l'on a besoin d'attacher des métadonnées aux nœuds.'''
    name: str

class BrandTaxonomy(BaseModel):
    '''Structure arborescente de la taxonomie : universe -> category -> subcategory -> product_type[]'''
    tree: dict[str, dict[str, dict[str, list[str]]]] = Field(
        default_factory=dict,
        description="Arbre taxonomique stocké sous forme de dictionnaire imbriqué"
    )

    def get_universes(self) -> list[str]:
        '''Retourne la liste de tous les univers existants.'''
        return list(self.tree.keys())

    def get_categories(self, universe: str) -> list[str]:
        '''Retourne la liste des catégories pour un univers donné.'''
        if universe in self.tree:
            return list(self.tree[universe].keys())
        return []

    def is_valid_path(self, universe: str, category: str, subcategory: str, product_type: str) -> bool:
        '''Vérifie si le chemin taxonomique complet existe et est valide.'''
        try:
            valid_product_types = self.tree[universe][category][subcategory]
            return product_type in valid_product_types
        except KeyError:
            return False

    def serialize(self) -> str:
        '''Sérialise la taxonomie en chaîne JSON.'''
        return self.model_dump_json(indent=2)

    @classmethod
    def deserialize(cls, json_data: str | dict[str, Any]) -> BrandTaxonomy:
        '''Désérialise une taxonomie à partir d'une chaîne JSON ou d'un dictionnaire.'''
        if isinstance(json_data, str):
            return cls.model_validate_json(json_data)
        return cls.model_validate(json_data)
