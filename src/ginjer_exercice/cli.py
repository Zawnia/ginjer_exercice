from pathlib import Path
import os
import sqlite3

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


@app.command()
def refresh_taxonomy(
    brand: str = typer.Option("ALL", help="Marque à rafraîchir, ou ALL pour la canonique."),
) -> None:
    """Re-parse le fichier JSON et régénère la taxonomie."""
    if brand == "ALL":
        typer.echo("Rafraîchissement de la taxonomie canonique depuis product_categorisation.json...")
        taxo = parse_canonical_taxonomy(Path("product_categorisation.json"))
        store = TaxonomyStore()
        store.save_taxonomy("canonical", taxo)
        typer.echo(f"Taxonomie canonique rafraîchie avec succès: {len(taxo.get_universes())} univers trouvés.")
        return

    try:
        target_brand = Brand(brand)
        load_taxonomy(target_brand, force_refresh=True)
        typer.echo(f"Taxonomie rafraîchie pour {target_brand.value}.")
    except ValueError:
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

    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

    bq_client = BigQueryClient(
        bq_client=bigquery.Client(project=settings.gcp_project_id or "ginjer-440122")
    )
    llm_provider = get_provider(
        settings.llm_provider,
        use_vertex=settings.llm_provider.lower() == "gemini" and settings.gcp_project_id is not None,
        project_id=settings.gcp_project_id,
        api_key=settings.openai_api_key,
    )
    prompt_registry = PromptRegistry()
    repository = SQLiteResultsRepository(sqlite3.connect(resolved_db_path))

    ad = bq_client.fetch_ad(ad_id)
    output = run_ad(
        ad,
        llm_provider=llm_provider,
        prompt_registry=prompt_registry,
        results_repository=repository,
    )

    named_products = sum(1 for product in output.products if product.name_info and product.name_info.name)
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
