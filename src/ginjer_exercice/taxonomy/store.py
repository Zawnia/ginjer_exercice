import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.ginjer_exercice.exceptions import TaxonomyNotFoundError
from src.ginjer_exercice.schemas.taxonomy import BrandTaxonomy


class TaxonomyStore:
    """Gère la persistence et le chargement des taxonomies sur disque.

    Les fichiers sont stockés dans ``data_dir`` sous la forme ``{name}.json``.
    Chaque fichier embarque des métadonnées d'auditabilité (source, timestamp).

    Args:
        data_dir: Répertoire de stockage. Créé automatiquement s'il n'existe pas.
    """

    def __init__(self, data_dir: str | Path = "data/taxonomies"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_taxonomy(self, name: str, taxonomy: BrandTaxonomy, source: str = "canonical") -> None:
        """Sauvegarde une taxonomie en JSON avec ses métadonnées d'auditabilité.

        Args:
            name: Identifiant logique (ex: ``"canonical"``, ``"CHANEL"``).
            taxonomy: Instance ``BrandTaxonomy`` à persister.
            source: Provenance de la taxonomie (chemin du fichier source, URL, etc.).
        """
        file_path = self.data_dir / f"{name.lower()}.json"

        payload = {
            "name": name,
            "source": source,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "is_canonical": name.lower() == "canonical",
            "taxonomy": taxonomy.model_dump(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_taxonomy(self, name: str) -> BrandTaxonomy:
        """Charge une taxonomie depuis le fichier JSON persisté.

        Args:
            name: Identifiant logique du fichier à charger.

        Returns:
            Une instance ``BrandTaxonomy`` validée.

        Raises:
            TaxonomyNotFoundError: Si le fichier n'existe pas sur disque.
            TaxonomyNotFoundError: Si la structure de l'enveloppe JSON est invalide.
        """
        file_path = self.data_dir / f"{name.lower()}.json"

        if not file_path.exists():
            raise TaxonomyNotFoundError(f"Aucune taxonomie persistée pour '{name}' dans {self.data_dir}")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise TaxonomyNotFoundError(
                    f"Le fichier de taxonomie '{name}' est corrompu (JSON invalide): {exc}"
                ) from exc

        if not isinstance(data, dict) or "taxonomy" not in data:
            raise TaxonomyNotFoundError(
                f"L'enveloppe du fichier de taxonomie '{name}' est invalide (clé 'taxonomy' manquante)"
            )

        try:
            return BrandTaxonomy.model_validate(data["taxonomy"])
        except ValidationError as exc:
            raise TaxonomyNotFoundError(
                f"La taxonomie persistée pour '{name}' ne valide pas le schéma Pydantic: {exc}"
            ) from exc
