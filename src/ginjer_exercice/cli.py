import typer
from src.ginjer_exercice.schemas.ad import Brand
from src.ginjer_exercice.taxonomy.loader import load_taxonomy

app = typer.Typer(help="CLI pour le pipeline de détection de produits.")

@app.command()
def refresh_taxonomy(brand: str = typer.Option("ALL", help="Marque à rafraîchir, ou ALL pour la canonique.")):
    """Re-parse le fichier JSON et regénère la taxonomie."""
    
    if brand == "ALL":
        typer.echo("Rafraîchissement de la taxonomie canonique depuis product_categorisation.json...")
        from src.ginjer_exercice.taxonomy.product_categorisation_parser import parse_canonical_taxonomy
        from src.ginjer_exercice.taxonomy.store import TaxonomyStore
        from pathlib import Path
        
        taxo = parse_canonical_taxonomy(Path("product_categorisation.json"))
        store = TaxonomyStore()
        store.save_taxonomy("canonical", taxo)
        typer.echo(f"Taxonomie canonique rafraîchie avec succès: {len(taxo.get_universes())} univers trouvés.")
    else:
        try:
            b = Brand(brand)
            taxo = load_taxonomy(b, force_refresh=True)
            typer.echo(f"Taxonomie rafraîchie pour {b.value}.")
        except ValueError:
            typer.echo(f"Erreur : Marque {brand} non supportée.", err=True)
            raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
