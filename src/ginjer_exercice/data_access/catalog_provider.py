"""Catalog providers used by Step 5 fallback enrichment."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..schemas.ad import Brand

_DEFAULT_CANONICAL_TAXONOMY_PATH = Path("data/taxonomies/canonical.json")


class CatalogProvider(ABC):
    """Source of reference catalog entries filtered by brand and taxonomy."""

    @abstractmethod
    def get_subset(self, brand: Brand, universe: str, category: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return a filtered catalog subset for Step 5 prompting.

        Args:
            brand: Brand currently being processed.
            universe: Product universe from taxonomy classification.
            category: Product category from taxonomy classification.
            limit: Maximum number of entries returned.

        Returns:
            A list of serializable dictionaries suitable for prompt injection.
        """


class CanonicalTaxonomyCatalogProvider(CatalogProvider):
    """Expose canonical taxonomy entries as lightweight catalog references."""

    def __init__(self, taxonomy_path: Path = _DEFAULT_CANONICAL_TAXONOMY_PATH) -> None:
        """Initialize the provider with a persisted taxonomy file.

        Args:
            taxonomy_path: Path to the persisted canonical taxonomy JSON file.
        """
        self._taxonomy_path = Path(taxonomy_path)

    def get_subset(self, brand: Brand, universe: str, category: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return flattened taxonomy entries filtered by universe and category.

        Args:
            brand: Brand currently being processed.
            universe: Product universe from taxonomy classification.
            category: Product category from taxonomy classification.
            limit: Maximum number of entries returned.

        Returns:
            A list of dictionaries with taxonomy reference entries.
        """
        if limit <= 0:
            return []

        payload = self._load_taxonomy_payload()
        tree = payload.get("taxonomy", {}).get("tree", {})
        categories = tree.get(universe, {})
        subcategories = categories.get(category, [])

        subset: list[dict[str, Any]] = []
        for subcategory in subcategories[:limit]:
            subset.append(
                {
                    "brand": brand.value,
                    "universe": universe,
                    "category": category,
                    "subcategory": subcategory,
                }
            )
        return subset

    def _load_taxonomy_payload(self) -> dict[str, Any]:
        """Load the persisted taxonomy envelope from disk.

        Returns:
            The raw JSON payload loaded from disk.

        Raises:
            FileNotFoundError: If the taxonomy file does not exist.
            ValueError: If the file content is not a JSON object.
        """
        if not self._taxonomy_path.exists():
            raise FileNotFoundError(f"Canonical taxonomy file not found: {self._taxonomy_path}")

        with self._taxonomy_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid taxonomy payload in {self._taxonomy_path}")

        return payload


class NullCatalogProvider(CatalogProvider):
    """Fallback provider used when no reference catalog is available."""

    def get_subset(self, brand: Brand, universe: str, category: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return an empty catalog subset.

        Args:
            brand: Brand currently being processed.
            universe: Product universe from taxonomy classification.
            category: Product category from taxonomy classification.
            limit: Maximum number of entries requested.

        Returns:
            Always an empty list.
        """
        return []


def get_catalog_provider(brand: Brand) -> CatalogProvider:
    """Return the catalog provider configured for a brand.

    Args:
        brand: Brand currently being processed.

    Returns:
        A canonical-taxonomy-backed provider for CHANEL, else a null provider.
    """
    if brand == Brand.CHANEL:
        return CanonicalTaxonomyCatalogProvider()
    return NullCatalogProvider()
