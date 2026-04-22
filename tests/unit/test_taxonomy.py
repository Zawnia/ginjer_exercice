import json
import pytest
from pathlib import Path
from datetime import datetime

from src.ginjer_exercice.schemas.taxonomy import BrandTaxonomy, NO_SUBCATEGORY_SENTINEL
from src.ginjer_exercice.schemas.ad import Brand
from src.ginjer_exercice.exceptions import TaxonomyNotFoundError
from src.ginjer_exercice.taxonomy.store import TaxonomyStore
from src.ginjer_exercice.taxonomy.loader import load_taxonomy
from src.ginjer_exercice.taxonomy.product_categorisation_parser import parse_canonical_taxonomy




def _make_schema(any_of_branches: list) -> dict:
    return {
        "schema": {
            "properties": {
                "products": {
                    "items": {
                        "properties": {
                            "product_categorisation": {
                                "anyOf": any_of_branches
                            }
                        }
                    }
                }
            }
        }
    }



@pytest.fixture
def dummy_json_schema(tmp_path: Path) -> Path:
    """Schéma minimal : une branche standard + une branche sans subcategory."""
    schema = _make_schema([
        {
            "properties": {
                "universe": {"enum": ["Women", "Men"]},
                "category": {"enum": ["Accessories"]},
                "subcategory": {"enum": ["Belts", "Hats"]},
            }
        },
        {
            "properties": {
                "universe": {"enum": ["Unisex"]},
                "category": {"enum": ["Art & Culture"]},
                # pas de subcategory — catégorie terminale
            }
        },
    ])
    file_path = tmp_path / "product_categorisation.json"
    file_path.write_text(json.dumps(schema), encoding="utf-8")
    return file_path


class TestParseCanonicalTaxonomy:

    def test_parses_standard_branch(self, dummy_json_schema: Path):
        taxo = parse_canonical_taxonomy(dummy_json_schema)
        assert "Women" in taxo.get_universes()
        assert "Men" in taxo.get_universes()
        assert set(taxo.get_categories("Women")) == {"Accessories"}
        assert set(taxo.get_subcategories("Women", "Accessories")) == {"Belts", "Hats"}

    def test_branch_without_subcategory_uses_sentinel(self, dummy_json_schema: Path):
        taxo = parse_canonical_taxonomy(dummy_json_schema)
        subcats = taxo.get_subcategories("Unisex", "Art & Culture")
        assert NO_SUBCATEGORY_SENTINEL in subcats

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_canonical_taxonomy(tmp_path / "nonexistent.json")

    def test_invalid_structure_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"wrong_key": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="Structure inattendue"):
            parse_canonical_taxonomy(bad)

    def test_subcategories_are_sorted(self, dummy_json_schema: Path):
        taxo = parse_canonical_taxonomy(dummy_json_schema)
        subcats = taxo.get_subcategories("Women", "Accessories")
        assert subcats == sorted(subcats)

    def test_non_dict_anyof_branch_is_skipped(self, tmp_path: Path):
        """Une branche anyOf non-dict doit être ignorée sans crasher."""
        schema = _make_schema([
            "not a dict branch",  # invalide — doit être ignoré
            {
                "properties": {
                    "universe": {"enum": ["Women"]},
                    "category": {"enum": ["Bags"]},
                    "subcategory": {"enum": ["Handbags"]},
                }
            },
        ])
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(schema), encoding="utf-8")
        taxo = parse_canonical_taxonomy(f)
        assert "Women" in taxo.get_universes()
        assert "Handbags" in taxo.get_subcategories("Women", "Bags")

    def test_branch_with_empty_universe_enum_produces_no_entry(self, tmp_path: Path):
        """Une branche avec universe enum vide ne doit rien créer dans l'arbre."""
        schema = _make_schema([
            {
                "properties": {
                    "universe": {"enum": []},      # vide
                    "category": {"enum": ["Bags"]},
                    "subcategory": {"enum": ["Handbags"]},
                }
            },
        ])
        f = tmp_path / "empty_enum.json"
        f.write_text(json.dumps(schema), encoding="utf-8")
        taxo = parse_canonical_taxonomy(f)
        assert taxo.get_universes() == []



class TestBrandTaxonomy:

    def test_get_universes_is_sorted(self):
        taxo = BrandTaxonomy(tree={
            "Women": {"Bags": ["Handbags"]},
            "Beauty": {"Perfume": ["Cologne"]},
        })
        universes = taxo.get_universes()
        assert universes == sorted(universes)

    def test_get_categories_is_sorted(self):
        taxo = BrandTaxonomy(tree={
            "Women": {
                "Shoes": ["Heels"],
                "Bags": ["Handbags"],
            }
        })
        cats = taxo.get_categories("Women")
        assert cats == sorted(cats)

    def test_is_terminal_category_true(self):
        taxo = BrandTaxonomy(tree={"Unisex": {"Art & Culture": [NO_SUBCATEGORY_SENTINEL]}})
        assert taxo.is_terminal_category("Unisex", "Art & Culture") is True

    def test_is_terminal_category_false(self):
        taxo = BrandTaxonomy(tree={"Women": {"Bags": ["Handbags", "Totes"]}})
        assert taxo.is_terminal_category("Women", "Bags") is False

    def test_is_valid_path_rejects_sentinel(self):
        taxo = BrandTaxonomy(tree={"Unisex": {"Art & Culture": [NO_SUBCATEGORY_SENTINEL]}})
        assert taxo.is_valid_path("Unisex", "Art & Culture", NO_SUBCATEGORY_SENTINEL) is False

    def test_is_valid_path_accepts_real_subcategory(self):
        taxo = BrandTaxonomy(tree={"Women": {"Bags": ["Handbags"]}})
        assert taxo.is_valid_path("Women", "Bags", "Handbags") is True



class TestTaxonomyStore:

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        taxo = BrandTaxonomy(tree={"Women": {"Accessories": ["Belts"]}})
        store.save_taxonomy("canonical", taxo, source="test")
        loaded = store.load_taxonomy("canonical")
        assert loaded.get_universes() == ["Women"]

    def test_save_writes_metadata(self, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        taxo = BrandTaxonomy(tree={"Beauty": {"Perfume": ["Women's Perfume"]}})
        store.save_taxonomy("canonical", taxo, source="data/raw/product_categorisation.json")
        data = json.loads((tmp_path / "canonical.json").read_text())
        assert "generated_at" in data
        assert data["source"] == "data/raw/product_categorisation.json"
        assert data["name"] == "canonical"
        assert data["is_canonical"] is True
        datetime.fromisoformat(data["generated_at"])  # valide ISO 8601

    def test_load_missing_raises_taxonomy_not_found(self, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        with pytest.raises(TaxonomyNotFoundError, match="Aucune taxonomie"):
            store.load_taxonomy("unknown")

    def test_load_corrupted_json_raises_taxonomy_not_found(self, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        (tmp_path / "bad.json").write_text("{invalid json", encoding="utf-8")
        with pytest.raises(TaxonomyNotFoundError, match="corrompu"):
            store.load_taxonomy("bad")

    def test_load_missing_taxonomy_key_raises(self, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        (tmp_path / "broken.json").write_text(json.dumps({"source": "x"}), encoding="utf-8")
        with pytest.raises(TaxonomyNotFoundError, match="clé 'taxonomy' manquante"):
            store.load_taxonomy("broken")


class TestLoadTaxonomy:

    def test_returns_canonical_when_no_brand_specific(self, dummy_json_schema: Path, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        taxo = load_taxonomy(Brand.CHANEL, store=store, canonical_source_path=dummy_json_schema)
        assert "Women" in taxo.get_universes()

    def test_persists_canonical_after_bootstrap(self, dummy_json_schema: Path, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        load_taxonomy(Brand.DIOR, store=store, canonical_source_path=dummy_json_schema)
        loaded = store.load_taxonomy("canonical")
        assert "Women" in loaded.get_universes()

    def test_force_refresh_bypasses_cache(self, dummy_json_schema: Path, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        load_taxonomy(Brand.CHANEL, store=store, canonical_source_path=dummy_json_schema)
        taxo = load_taxonomy(Brand.CHANEL, force_refresh=True, store=store, canonical_source_path=dummy_json_schema)
        assert "Women" in taxo.get_universes()

    def test_missing_source_raises_file_not_found(self, tmp_path: Path):
        store = TaxonomyStore(data_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            load_taxonomy(Brand.MFK, store=store, canonical_source_path=tmp_path / "nonexistent.json")

    def test_only_catches_taxonomy_not_found_not_other_errors(self, tmp_path: Path):
        """Les exceptions non liées à l'absence de fichier doivent remonter."""
        store = TaxonomyStore(data_dir=tmp_path)
        bad = tmp_path / "bad_schema.json"
        bad.write_text(json.dumps({"wrong": "structure"}), encoding="utf-8")
        with pytest.raises(ValueError, match="Structure inattendue"):
            load_taxonomy(Brand.LOUIS_VUITTON, store=store, canonical_source_path=bad)


class TestLoadTaxonomyBrandPriority:

    def test_brand_specific_takes_priority_over_canonical(self, dummy_json_schema: Path, tmp_path: Path):
        """Une taxonomie spécialisée marque doit primer sur la canonique."""
        store = TaxonomyStore(data_dir=tmp_path)

        generic = BrandTaxonomy(tree={"Beauty": {"Perfume": ["Cologne"]}})
        store.save_taxonomy("canonical", generic)

        chanel_specific = BrandTaxonomy(tree={"Women": {"Bags": ["2.55", "Classic Flap"]}})
        store.save_taxonomy(Brand.CHANEL.value, chanel_specific)

        taxo = load_taxonomy(Brand.CHANEL, store=store, canonical_source_path=dummy_json_schema)

        assert "Women" in taxo.get_universes()
        assert "Beauty" not in taxo.get_universes()
        assert "2.55" in taxo.get_subcategories("Women", "Bags")

    def test_falls_back_to_canonical_when_no_brand_file(self, dummy_json_schema: Path, tmp_path: Path):
        """Sans fichier marque, la canonique persistée doit être renvoyée."""
        store = TaxonomyStore(data_dir=tmp_path)
        generic = BrandTaxonomy(tree={"Beauty": {"Perfume": ["Cologne"]}})
        store.save_taxonomy("canonical", generic)

        taxo = load_taxonomy(Brand.CHANEL, store=store, canonical_source_path=dummy_json_schema)
        assert "Beauty" in taxo.get_universes()
        assert "Women" not in taxo.get_universes()
