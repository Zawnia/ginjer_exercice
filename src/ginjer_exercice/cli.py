from pathlib import Path
import logging
import os
import sqlite3
import sys

import typer
from google.cloud import bigquery

from ginjer_exercice.config import get_settings
from ginjer_exercice.data_access.bigquery_client import BigQueryClient
from ginjer_exercice.data_access.results_repository import SQLiteResultsRepository
from ginjer_exercice.llm.factory import get_provider
from ginjer_exercice.observability.prompts import PromptRegistry
from ginjer_exercice.pipeline.orchestrator import run_ad
from ginjer_exercice.schemas.ad import Brand
from ginjer_exercice.taxonomy.loader import load_taxonomy
from ginjer_exercice.taxonomy.product_categorisation_parser import parse_canonical_taxonomy
from ginjer_exercice.taxonomy.store import TaxonomyStore

app = typer.Typer(help="CLI pour le pipeline de détection de produits.")
logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_level(*, verbose: bool, debug: bool) -> int:
    if debug:
        return logging.DEBUG
    if verbose:
        return logging.INFO
    return logging.WARNING


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    _configure_console_encoding()
    level = _resolve_log_level(verbose=verbose, debug=debug)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING if not debug else logging.INFO)
    logging.getLogger("google").setLevel(logging.WARNING if not debug else logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING if not debug else logging.INFO)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", help="Active les logs INFO du pipeline."),
    debug: bool = typer.Option(False, "--debug", help="Active les logs DEBUG détaillés."),
) -> None:
    """Configure les options globales de la CLI avant l'exécution des commandes."""
    _configure_logging(verbose=verbose, debug=debug)


@app.command()
def refresh_taxonomy(
    brand: str = typer.Option("ALL", help="Marque à rafraîchir, ou ALL pour la canonique."),
) -> None:
    """Re-parse le fichier JSON et régénère la taxonomie."""
    logger.info("refresh-taxonomy started for brand=%s", brand)
    if brand == "ALL":
        typer.echo("Rafraîchissement de la taxonomie canonique depuis product_categorisation.json...")
        taxo = parse_canonical_taxonomy(Path("product_categorisation.json"))
        store = TaxonomyStore()
        store.save_taxonomy("canonical", taxo)
        logger.info("Canonical taxonomy refreshed with %d universes", len(taxo.get_universes()))
        typer.echo(f"Taxonomie canonique rafraîchie avec succès: {len(taxo.get_universes())} univers trouvés.")
        return

    try:
        target_brand = Brand(brand)
        load_taxonomy(target_brand, force_refresh=True)
        logger.info("Brand taxonomy refreshed for %s", target_brand.value)
        typer.echo(f"Taxonomie rafraîchie pour {target_brand.value}.")
    except ValueError:
        logger.error("Unsupported brand provided to refresh-taxonomy: %s", brand)
        typer.echo(f"Erreur : Marque {brand} non supportée.", err=True)
        raise typer.Exit(code=1)


@app.command("process-ad")
def process_ad(
    ad_id: str = typer.Argument(..., help="ID BigQuery de la publicité à traiter."),
    db_path: str | None = typer.Option(None, "--db-path", help="Chemin SQLite de persistance."),
) -> None:
    """Traite une pub par `ad_id`, persiste le résultat et affiche un résumé."""
    settings = get_settings()
    resolved_db_path = Path(db_path or settings.sqlite_db_path)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("process-ad started for ad_id=%s db_path=%s", ad_id, resolved_db_path)

    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        logger.debug("GOOGLE_APPLICATION_CREDENTIALS configured from settings")

    project_id = settings.gcp_project_id or "ginjer-440122"
    logger.info("Initializing BigQuery client for project=%s", project_id)
    bq_client = BigQueryClient(
        bq_client=bigquery.Client(project=project_id)
    )
    logger.info("Initializing provider=%s", settings.llm_provider)
    llm_provider = get_provider(
        settings.llm_provider,
        use_vertex=settings.llm_provider.lower() == "gemini" and settings.gcp_project_id is not None,
        project_id=settings.gcp_project_id,
        api_key=settings.openai_api_key,
    )
    prompt_registry = PromptRegistry()
    repository = SQLiteResultsRepository(sqlite3.connect(resolved_db_path))
    logger.info("Fetching ad_id=%s from BigQuery", ad_id)

    ad = bq_client.fetch_ad(ad_id)
    logger.info(
        "Ad fetched: ad_id=%s brand=%s texts=%d media=%d",
        ad.platform_ad_id,
        ad.brand.value,
        len(ad.texts),
        len(ad.media_urls),
    )
    output = run_ad(
        ad,
        llm_provider=llm_provider,
        prompt_registry=prompt_registry,
        results_repository=repository,
    )

    named_products = sum(1 for product in output.products if product.name_info and product.name_info.name)
    logger.info(
        "process-ad completed: ad_id=%s products=%d names_found=%d needs_review=%s trace_id=%s",
        output.ad_id,
        len(output.products),
        named_products,
        output.needs_review,
        output.trace_id,
    )
    typer.echo(
        "\n".join(
            [
                f"ad_id: {output.ad_id}",
                f"brand: {output.brand.value}",
                f"products: {len(output.products)}",
                f"names_found: {named_products}",
                f"needs_review: {output.needs_review}",
                f"trace_id: {output.trace_id}",
                f"sqlite_path: {resolved_db_path}",
            ]
        )
    )


if __name__ == "__main__":
    app()
