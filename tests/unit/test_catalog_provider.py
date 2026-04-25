from __future__ import annotations

import json
import shutil
from pathlib import Path

from ginjer_exercice.data_access.catalog_provider import (
    CanonicalTaxonomyCatalogProvider,
    NullCatalogProvider,
)
from ginjer_exercice.schemas.ad import Brand


def test_canonical_taxonomy_catalog_provider_filters_subset() -> None:
    test_dir = Path("output/test_catalog_provider")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_path = test_dir / "canonical.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "taxonomy": {
                    "tree": {
                        "Beauty": {
                            "Perfume": ["Women's Perfume", "Cologne"],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    provider = CanonicalTaxonomyCatalogProvider(taxonomy_path=taxonomy_path)

    subset = provider.get_subset(Brand.CHANEL, universe="Beauty", category="Perfume")

    assert subset == [
        {"brand": "CHANEL", "universe": "Beauty", "category": "Perfume", "subcategory": "Women's Perfume"},
        {"brand": "CHANEL", "universe": "Beauty", "category": "Perfume", "subcategory": "Cologne"},
    ]


def test_null_catalog_provider_returns_empty_list() -> None:
    provider = NullCatalogProvider()

    assert provider.get_subset(Brand.CHANEL, universe="Beauty", category="Perfume") == []
