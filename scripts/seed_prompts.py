"""Pousse les prompts YAML locaux vers Langfuse.

Utilisation :
    uv run python scripts/seed_prompts.py
    uv run python scripts/seed_prompts.py --dry-run
"""
import yaml
from pathlib import Path
import typer

from src.ginjer_exercice.observability.client import get_langfuse_client

app = typer.Typer(help="CLI pour synchroniser les prompts locaux avec Langfuse.")

@app.command()
def seed(dry_run: bool = typer.Option(False, "--dry-run", help="Ne pas envoyer les données à Langfuse.")):
    """Pousse les prompts YAML de /prompts vers Langfuse."""
    
    langfuse = get_langfuse_client()
    if langfuse is None and not dry_run:
        typer.echo("Erreur: Langfuse non configuré. Impossible de seeder.", err=True)
        raise typer.Exit(code=1)
        
    base_dir = Path(__file__).resolve().parent.parent
    prompts_dir = base_dir / "prompts"
    
    if not prompts_dir.exists():
        typer.echo(f"Erreur: Le dossier {prompts_dir} n'existe pas.", err=True)
        raise typer.Exit(code=1)
        
    yaml_files = list(prompts_dir.glob("*.yaml"))
    typer.echo(f"Trouvé {len(yaml_files)} fichiers YAML de prompts.")
    
    for filepath in yaml_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                
            name = data.get("name")
            label = data.get("label", "production")
            prompt_text = data.get("prompt", "")
            config = data.get("config", {})
            type_ = data.get("type", "text")
            
            if not name:
                typer.echo(f"  [SKIPPED] {filepath.name}: Pas de champ 'name'.", err=True)
                continue
                
            if dry_run:
                typer.echo(f"  [DRY-RUN] Préparation du prompt '{name}' (label: {label})")
            else:
                langfuse.create_prompt(
                    name=name,
                    prompt=prompt_text,
                    config=config,
                    labels=[label],
                    type=type_
                )
                typer.echo(f"  [OK] Prompt '{name}' poussé avec succès.")
                
        except Exception as e:
            typer.echo(f"  [ERREUR] Impossible de traiter {filepath.name}: {e}", err=True)

if __name__ == "__main__":
    app()
