"""Smoke test end-to-end du flux `process-ad` avec image multimodale.

Objectif :
- exercer la vraie CLI `process-ad`
- faire passer par le vrai orchestrateur
- exécuter les vrais steps 1-4
- valider que `step2` reçoit bien une image binaire
- vérifier que le résultat final est persisté en SQLite

Le script remplace uniquement les frontières externes :
- BigQuery
- provider LLM
- taxonomie
- fetcher média

Il ne dépend ni d'un vrai réseau ni d'un vrai provider.

Usage :
    python scripts/e2e_process_ad_cli_multimodal.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel
from typer.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ginjer_exercice.cli import app
from ginjer_exercice.data_access.results_repository import SQLiteResultsRepository
from ginjer_exercice.llm.base import LLMCallConfig, LLMMessage, LLMProvider, LLMResponse, TraceContext
from ginjer_exercice.observability.prompts import ManagedPrompt
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.media import MediaContent, MediaKind
from ginjer_exercice.schemas.products import ProductClassification
from ginjer_exercice.schemas.step_outputs import (
    DetectedProductLLM,
    DetectedProductList,
    ExtractedName,
    UniverseDetection,
    UniverseResult,
)


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(encoding, errors="replace"))


class SequenceLLMProvider(LLMProvider):
    """Provider fake séquentiel pour dérouler les vrais steps."""

    def __init__(self, canned_responses: list[BaseModel]) -> None:
        self._responses = deque(canned_responses)
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "SequenceLLMProvider"

    @property
    def supports_video(self) -> bool:
        return False

    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        config: LLMCallConfig,
        trace_context: TraceContext | None = None,
    ) -> LLMResponse:
        if not self._responses:
            raise AssertionError("Plus de réponses fake disponibles.")
        parsed = self._responses.popleft()
        self.calls.append(
            {
                "messages": messages,
                "response_model": response_model,
                "config": config,
            }
        )
        return LLMResponse(
            parsed=parsed,
            raw_json=parsed.model_dump_json(),
            usage=(123, 45),
            latency_ms=12,
            model_used="fake-e2e-model",
        )


class FakePromptRegistry:
    """Registre minimal pour les vrais steps, sans dépendance Langfuse."""

    def __init__(self) -> None:
        self.gets: list[str] = []

    def get(self, name: str, label: str = "production") -> ManagedPrompt:
        self.gets.append(name)
        return ManagedPrompt(
            name=name,
            version="e2e-test",
            label=label,
            prompt=(
                "Brand: {{brand}}\n"
                "Text: {{texts_block}}\n"
                "Media: {{media_count}}\n"
                "Universes: {{universes}}\n"
                "Product: {{product_description}}\n"
                "Color: {{product_color}}\n"
                "Universe: {{product_universe}}\n"
                "Category: {{product_category}}\n"
                "Subcategory: {{product_subcategory}}\n"
                "Taxonomy: {{taxonomy_tree}}\n"
            ),
            config={
                "model": "fake-e2e-model",
                "temperature": 0.0,
                "max_tokens": 4000,
            },
            source="yaml_fallback",
        )


class FakeBigQueryClient:
    def __init__(self, ad: Ad) -> None:
        self._ad = ad

    def fetch_ad(self, ad_id: str) -> Ad:
        if ad_id != self._ad.platform_ad_id:
            raise ValueError(f"Ad inconnue: {ad_id}")
        return self._ad


class FakeTaxonomy:
    """Taxonomie minimale compatible avec step3 et le scoring."""

    def slice_for_universe(self, universe: str) -> "FakeTaxonomy":
        return self

    def format_as_bullet_list(self) -> str:
        return "- Fragrance\n  - Perfume\n    - Eau de parfum\n"

    def is_terminal_category(self, universe: str, category: str) -> bool:
        return False

    def is_valid_path(self, universe: str, category: str, subcategory: str) -> bool:
        return (
            universe == "Fragrance"
            and category == "Perfume"
            and subcategory == "Eau de parfum"
        )

    def list_valid_subcategories(self, universe: str, category: str) -> list[str]:
        if universe == "Fragrance" and category == "Perfume":
            return ["Eau de parfum"]
        return []


class FakeMediaFetcher:
    def __init__(self, image_url: str) -> None:
        self.image_url = image_url
        self.downloaded_urls: list[str] = []

    def download(self, url: str) -> MediaContent:
        self.downloaded_urls.append(url)
        if url != self.image_url:
            raise ValueError(f"URL inattendue: {url}")
        content = b"fake-image-binary"
        return MediaContent(
            url=url,
            kind=MediaKind.IMAGE,
            mime_type="image/jpeg",
            content=content,
            size_bytes=len(content),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test E2E du flux process-ad avec multimodal image."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active les logs INFO de la CLI et du pipeline pendant le smoke test.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active les logs DEBUG détaillés de la CLI et du pipeline.",
    )
    args = parser.parse_args()

    runner = CliRunner()

    image_url = "https://example.com/chanel-n5.jpg"
    ad = Ad(
        platform_ad_id="e2e-ad-001",
        brand=Brand.CHANEL,
        texts=[
            AdText(
                title="CHANEL N5",
                body_text="Legendary fragrance campaign.",
                caption="Bottle close-up.",
            )
        ],
        media_urls=[image_url],
    )

    fake_provider = SequenceLLMProvider(
        [
            UniverseResult(
                detected_universes=[
                    UniverseDetection(
                        universe="Fragrance",
                        confidence=0.96,
                        reasoning="Bottle and fragrance cues are visible.",
                    )
                ]
            ),
            DetectedProductList(
                products=[
                    DetectedProductLLM(
                        raw_description="Transparent rectangular perfume bottle with light label",
                        universe="Fragrance",
                        color="Gold",
                        importance=5,
                    )
                ],
                overall_confidence=0.93,
            ),
            ProductClassification(
                universe="Fragrance",
                category="Perfume",
                subcategory="Eau de parfum",
                confidence=0.88,
            ),
            ExtractedName(
                name="CHANEL N5",
                found_in="image_text",
                confidence=0.91,
            ),
        ]
    )

    fake_prompt_registry = FakePromptRegistry()
    fake_media_fetcher = FakeMediaFetcher(image_url=image_url)
    fake_taxonomy = FakeTaxonomy()

    db_path = PROJECT_ROOT / "output" / "e2e_process_ad_cli_multimodal.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    fake_settings = SimpleNamespace(
        sqlite_db_path=str(db_path),
        google_application_credentials=None,
        gcp_project_id=None,
        llm_provider="gemini",
        openai_api_key=None,
    )

    with (
        patch("ginjer_exercice.cli.get_settings", return_value=fake_settings),
        patch("ginjer_exercice.cli.bigquery.Client", return_value=object()),
        patch("ginjer_exercice.cli.BigQueryClient", return_value=FakeBigQueryClient(ad)),
        patch("ginjer_exercice.cli.get_provider", return_value=fake_provider),
        patch("ginjer_exercice.cli.PromptRegistry", return_value=fake_prompt_registry),
        patch("ginjer_exercice.pipeline.orchestrator.load_taxonomy", return_value=fake_taxonomy),
        patch("ginjer_exercice.observability.scoring.load_taxonomy", return_value=fake_taxonomy),
        patch("ginjer_exercice.pipeline.step2_products._default_media_fetcher", return_value=fake_media_fetcher),
    ):
        cli_args: list[str] = []
        if args.debug:
            cli_args.append("--debug")
        elif args.verbose:
            cli_args.append("--verbose")
        cli_args.extend(["process-ad", ad.platform_ad_id, "--db-path", str(db_path)])
        result = runner.invoke(app, cli_args)

    _safe_print("=== CLI STDOUT ===")
    if result.stdout.strip():
        _safe_print(result.stdout.strip())
    else:
        _safe_print("(empty)")
    stderr_output = getattr(result, "stderr", "")
    if stderr_output and stderr_output.strip():
        _safe_print("")
        _safe_print("=== CLI STDERR ===")
        _safe_print(stderr_output.strip())
    _safe_print("")

    if result.exit_code != 0:
        raise SystemExit(f"Echec CLI, code={result.exit_code}")

    if len(fake_provider.calls) != 4:
        raise SystemExit(f"Nombre d'appels LLM inattendu: {len(fake_provider.calls)} au lieu de 4")

    step2_call = fake_provider.calls[1]
    step2_parts = step2_call["messages"][0].parts
    has_binary_image = any(
        getattr(part, "type", None) == "media" and isinstance(getattr(part, "media", None), bytes)
        for part in step2_parts
    )
    if not has_binary_image:
        raise SystemExit("Step2 n'a pas reçu d'image binaire dans le message multimodal.")

    if fake_media_fetcher.downloaded_urls != [image_url]:
        raise SystemExit(f"MediaFetcher non appelé comme attendu: {fake_media_fetcher.downloaded_urls}")

    repository = SQLiteResultsRepository(sqlite3.connect(db_path))
    persisted = repository.get(ad.platform_ad_id)
    if persisted is None:
        raise SystemExit("Aucun résultat persisté en SQLite.")

    _safe_print("=== VALIDATION ===")
    _safe_print(f"ad_id: {persisted.ad_id}")
    _safe_print(f"brand: {persisted.brand.value}")
    _safe_print(f"products: {len(persisted.products)}")
    _safe_print(f"needs_review: {persisted.needs_review}")
    _safe_print(f"taxonomy_coherence: {persisted.scores.taxonomy_coherence}")
    _safe_print(f"confidence: {persisted.scores.confidence}")
    _safe_print(f"sqlite_path: {db_path}")
    _safe_print("")
    _safe_print("Smoke test E2E OK: CLI + orchestrateur + multimodal image + SQLite.")


if __name__ == "__main__":
    main()
