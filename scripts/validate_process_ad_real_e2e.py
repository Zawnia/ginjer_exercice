"""Real end-to-end validation for `process-ad`.

This script is intentionally stricter than `e2e_process_ad_cli_multimodal.py`.
It is designed to produce evidence that the built pipeline works with:

- real BigQuery reads
- real prompt loading
- real media downloads
- real LLM calls
- the real orchestrator
- the real CLI executed as a subprocess
- real SQLite persistence

Validation strategy:
1. Fetch one real ad, either by explicit `--ad-id` or by scanning a brand.
2. Preflight at least one downloadable image URL.
3. Run the orchestrator directly with a recording wrapper around the real LLM
   provider to prove that `step2` sent binary image data to the model.
4. Run the actual CLI in a separate Python process against the same ad.
5. Re-open both SQLite databases and validate persisted outputs.

Usage examples:
    python scripts/validate_process_ad_real_e2e.py --ad-id <platform_ad_id> --verbose
    python scripts/validate_process_ad_real_e2e.py --brand CHANEL --scan-limit 10 --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from google.cloud import bigquery
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ginjer_exercice.config import get_settings
from ginjer_exercice.data_access.bigquery_client import BigQueryClient
from ginjer_exercice.data_access.media_fetcher import MediaFetcher
from ginjer_exercice.data_access.results_repository import SQLiteResultsRepository
from ginjer_exercice.llm.base import LLMCallConfig, LLMMessage, LLMProvider, LLMResponse, TraceContext
from ginjer_exercice.llm.factory import get_provider
from ginjer_exercice.observability.prompts import PromptRegistry
from ginjer_exercice.pipeline.orchestrator import run_ad
from ginjer_exercice.schemas.ad import Ad, Brand
from ginjer_exercice.schemas.media import MediaKind

logger = logging.getLogger("validate_process_ad_real_e2e")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class MediaPreflight:
    url: str
    mime_type: str
    size_bytes: int


@dataclass
class RecordedLLMCall:
    response_model: str
    text_parts: int
    media_parts: int
    binary_media_parts: int


class RecordingLLMProvider(LLMProvider):
    """Thin wrapper around a real provider to record message payload shape."""

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.calls: list[RecordedLLMCall] = []

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def supports_video(self) -> bool:
        return self._inner.supports_video

    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        config: LLMCallConfig,
        trace_context: TraceContext | None = None,
    ) -> LLMResponse:
        text_parts = 0
        media_parts = 0
        binary_media_parts = 0
        for message in messages:
            for part in message.parts:
                part_type = getattr(part, "type", None)
                if part_type == "text":
                    text_parts += 1
                elif part_type == "media":
                    media_parts += 1
                    if isinstance(getattr(part, "media", None), bytes):
                        binary_media_parts += 1

        self.calls.append(
            RecordedLLMCall(
                response_model=response_model.__name__,
                text_parts=text_parts,
                media_parts=media_parts,
                binary_media_parts=binary_media_parts,
            )
        )
        return self._inner.generate_structured(
            messages=messages,
            response_model=response_model,
            config=config,
            trace_context=trace_context,
        )


def _resolve_log_level(*, verbose: bool, debug: bool) -> int:
    if debug:
        return logging.DEBUG
    if verbose:
        return logging.INFO
    return logging.WARNING


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    logging.basicConfig(
        level=_resolve_log_level(verbose=verbose, debug=debug),
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING if not debug else logging.INFO)
    logging.getLogger("google").setLevel(logging.WARNING if not debug else logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING if not debug else logging.INFO)


def _safe_print(text: str = "") -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(encoding, errors="replace"))


def _build_bigquery_client() -> BigQueryClient:
    settings = get_settings()
    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

    project_id = settings.gcp_project_id or "ginjer-440122"
    return BigQueryClient(bq_client=bigquery.Client(project=project_id))


def _build_media_fetcher() -> MediaFetcher:
    settings = get_settings()
    return MediaFetcher(
        client=httpx.Client(timeout=max(settings.media_image_timeout, settings.media_video_timeout)),
        max_size_bytes=settings.media_max_size_bytes,
        image_timeout=settings.media_image_timeout,
        video_timeout=settings.media_video_timeout,
        max_retries=settings.media_max_retries,
    )


def _build_provider() -> LLMProvider:
    settings = get_settings()
    return get_provider(
        settings.llm_provider,
        use_vertex=settings.llm_provider.lower() == "gemini" and settings.gcp_project_id is not None,
        project_id=settings.gcp_project_id,
        api_key=settings.openai_api_key,
    )


def _preflight_first_image(ad: Ad, fetcher: MediaFetcher) -> MediaPreflight:
    for url in ad.media_urls:
        if not url.startswith(("http://", "https://")):
            logger.info("Skipping non-http media url during preflight: %s", url)
            continue

        try:
            media = fetcher.download(url)
        except Exception as exc:
            logger.warning("Media preflight failed for %s: %s", url, exc)
            continue

        if media.kind != MediaKind.IMAGE:
            logger.info("Skipping non-image media during preflight: %s (%s)", url, media.mime_type)
            continue

        return MediaPreflight(
            url=url,
            mime_type=media.mime_type or "unknown",
            size_bytes=media.size_bytes,
        )

    raise RuntimeError(
        f"No downloadable image could be validated for ad_id={ad.platform_ad_id}. "
        "This ad cannot prove the multimodal image path."
    )


def _select_ad(
    *,
    bq_client: BigQueryClient,
    media_fetcher: MediaFetcher,
    ad_id: str | None,
    brand: Brand,
    scan_limit: int,
) -> tuple[Ad, MediaPreflight]:
    if ad_id:
        ad = bq_client.fetch_ad(ad_id)
        return ad, _preflight_first_image(ad, media_fetcher)

    ads = bq_client.fetch_ads_by_brand(brand, limit=scan_limit)
    if not ads:
        raise RuntimeError(f"No ads returned from BigQuery for brand={brand.value}.")

    for ad in ads:
        if not ad.media_urls:
            continue
        try:
            preflight = _preflight_first_image(ad, media_fetcher)
            return ad, preflight
        except Exception as exc:
            logger.info("Ad skipped during selection: ad_id=%s reason=%s", ad.platform_ad_id, exc)

    raise RuntimeError(
        f"No ad with a downloadable image was found in the first {scan_limit} ads for brand={brand.value}."
    )


def _sqlite_repo(db_path: Path) -> SQLiteResultsRepository:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    return SQLiteResultsRepository(sqlite3.connect(db_path))


def _parse_cli_stdout(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _sanitize_fragment(value: str) -> str:
    cleaned = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def _run_orchestrator_validation(
    *,
    ad: Ad,
    db_path: Path,
    min_products: int,
) -> tuple[object, RecordingLLMProvider]:
    prompt_registry = PromptRegistry()
    recording_provider = RecordingLLMProvider(_build_provider())
    repository = _sqlite_repo(db_path)

    start = time.monotonic()
    output = run_ad(
        ad,
        llm_provider=recording_provider,
        prompt_registry=prompt_registry,
        results_repository=repository,
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    step2_calls = [call for call in recording_provider.calls if call.response_model == "DetectedProductList"]
    if not step2_calls:
        raise RuntimeError("No step2 LLM call was recorded during the orchestrator run.")

    step2_call = step2_calls[0]
    if step2_call.binary_media_parts < 1:
        raise RuntimeError(
            "The orchestrator run reached step2 but did not send any binary image part to the LLM."
        )

    persisted = repository.get(ad.platform_ad_id)
    if persisted is None:
        raise RuntimeError("The orchestrator run completed but did not persist any SQLite result.")

    if len(persisted.products) < min_products:
        raise RuntimeError(
            f"Persisted orchestrator output has {len(persisted.products)} products; expected at least {min_products}."
        )

    _safe_print("=== ORCHESTRATOR RUN ===")
    _safe_print(f"ad_id: {output.ad_id}")
    _safe_print(f"brand: {output.brand.value}")
    _safe_print(f"products: {len(output.products)}")
    _safe_print(f"needs_review: {output.needs_review}")
    _safe_print(f"trace_id: {output.trace_id}")
    _safe_print(f"taxonomy_coherence: {output.scores.taxonomy_coherence}")
    _safe_print(f"confidence: {output.scores.confidence}")
    _safe_print(f"sqlite_path: {db_path}")
    _safe_print(f"duration_ms: {latency_ms}")
    _safe_print(f"step2_binary_media_parts: {step2_call.binary_media_parts}")
    _safe_print(f"llm_calls_recorded: {len(recording_provider.calls)}")
    _safe_print("")

    return output, recording_provider


def _run_cli_validation(
    *,
    ad_id: str,
    db_path: Path,
    verbose: bool,
    debug: bool,
    min_products: int,
) -> object:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    command = [sys.executable, "-m", "ginjer_exercice.cli"]
    if debug:
        command.append("--debug")
    elif verbose:
        command.append("--verbose")
    command.extend(["process-ad", ad_id, "--db-path", str(db_path)])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    logger.info("Running CLI subprocess: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    _safe_print("=== CLI SUBPROCESS ===")
    _safe_print(f"exit_code: {result.returncode}")
    _safe_print("stdout:")
    _safe_print(result.stdout.rstrip() or "(empty)")
    _safe_print("")
    _safe_print("stderr:")
    _safe_print(result.stderr.rstrip() or "(empty)")
    _safe_print("")

    if result.returncode != 0:
        raise RuntimeError(f"CLI subprocess failed with exit code {result.returncode}.")

    repository = SQLiteResultsRepository(sqlite3.connect(db_path))
    persisted = repository.get(ad_id)
    if persisted is None:
        raise RuntimeError("The CLI subprocess completed but no SQLite result was found.")

    if len(persisted.products) < min_products:
        raise RuntimeError(
            f"Persisted CLI output has {len(persisted.products)} products; expected at least {min_products}."
        )

    stdout_map = _parse_cli_stdout(result.stdout)
    required_keys = {"ad_id", "brand", "products", "needs_review", "trace_id", "sqlite_path"}
    missing_keys = sorted(required_keys - set(stdout_map))
    if missing_keys:
        raise RuntimeError(f"CLI stdout is missing expected keys: {missing_keys}")

    _safe_print("=== CLI VALIDATION ===")
    _safe_print(f"ad_id: {persisted.ad_id}")
    _safe_print(f"brand: {persisted.brand.value}")
    _safe_print(f"products: {len(persisted.products)}")
    _safe_print(f"needs_review: {persisted.needs_review}")
    _safe_print(f"taxonomy_coherence: {persisted.scores.taxonomy_coherence}")
    _safe_print(f"confidence: {persisted.scores.confidence}")
    _safe_print(f"sqlite_path: {db_path}")
    _safe_print("")

    return persisted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real end-to-end validation for process-ad with BigQuery, LLM, multimodal image, orchestrator, CLI and SQLite."
    )
    parser.add_argument("--ad-id", help="Explicit platform_ad_id to validate.")
    parser.add_argument(
        "--brand",
        choices=[brand.value for brand in Brand],
        default=Brand.CHANEL.value,
        help="Brand used when auto-selecting an ad.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=10,
        help="How many ads to scan when auto-selecting a candidate ad.",
    )
    parser.add_argument(
        "--min-products",
        type=int,
        default=1,
        help="Minimum number of persisted products required for both runs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logs.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logs.")
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose, debug=args.debug)

    settings = get_settings()
    provider_name = settings.llm_provider
    _safe_print("=== REAL E2E VALIDATION ===")
    _safe_print(f"provider: {provider_name}")
    _safe_print(f"requested_ad_id: {args.ad_id or '(auto)'}")
    _safe_print(f"brand: {args.brand}")
    _safe_print(f"scan_limit: {args.scan_limit}")
    _safe_print("")

    bq_client = _build_bigquery_client()
    media_fetcher = _build_media_fetcher()
    brand = Brand(args.brand)

    ad, preflight = _select_ad(
        bq_client=bq_client,
        media_fetcher=media_fetcher,
        ad_id=args.ad_id,
        brand=brand,
        scan_limit=args.scan_limit,
    )

    _safe_print("=== SELECTED AD ===")
    _safe_print(f"ad_id: {ad.platform_ad_id}")
    _safe_print(f"brand: {ad.brand.value}")
    _safe_print(f"texts: {len(ad.texts)}")
    _safe_print(f"media_urls: {len(ad.media_urls)}")
    _safe_print(f"validated_image_url: {preflight.url}")
    _safe_print(f"validated_image_mime: {preflight.mime_type}")
    _safe_print(f"validated_image_size_bytes: {preflight.size_bytes}")
    _safe_print("")

    output_dir = PROJECT_ROOT / "output" / "real_e2e"
    ad_fragment = _sanitize_fragment(ad.platform_ad_id)
    direct_db_path = output_dir / f"{ad_fragment}.direct.sqlite"
    cli_db_path = output_dir / f"{ad_fragment}.cli.sqlite"

    _run_orchestrator_validation(
        ad=ad,
        db_path=direct_db_path,
        min_products=args.min_products,
    )
    _run_cli_validation(
        ad_id=ad.platform_ad_id,
        db_path=cli_db_path,
        verbose=args.verbose,
        debug=args.debug,
        min_products=args.min_products,
    )

    _safe_print("=== RESULT ===")
    _safe_print("Real E2E validation OK: BigQuery + media download + real LLM + orchestrator + CLI subprocess + SQLite.")


if __name__ == "__main__":
    main()
