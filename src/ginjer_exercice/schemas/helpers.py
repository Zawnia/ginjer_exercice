"""Helpers de normalisation pour les schémas — partagés entre data_access et le domaine.

La normalisation des marques est centralisée ici pour éviter la duplication
entre le BigQueryClient et tout autre point d'entrée futur.
"""

from .ad import Brand
from ..exceptions import UnsupportedBrandError


_BRAND_ALIASES: dict[str, Brand] = {
    "chanel": Brand.CHANEL,
    "dior": Brand.DIOR,
    "louis_vuitton": Brand.LOUIS_VUITTON,
    "balenciaga": Brand.BALENCIAGA,
    "mfk": Brand.MFK,
    "louis vuitton": Brand.LOUIS_VUITTON,
    "louisvuitton": Brand.LOUIS_VUITTON,
    "lv": Brand.LOUIS_VUITTON,
    "maison francis kurkdjian": Brand.MFK,
    "francis kurkdjian": Brand.MFK,
}


def normalize_brand(raw: str) -> Brand:
    """Normalise une chaîne brute en ``Brand`` enum.

    La recherche est insensible à la casse et supporte les alias
    définis dans ``_BRAND_ALIASES``.

    Args:
        raw: Valeur brute de la marque (ex: depuis BigQuery).

    Returns:
        L'enum ``Brand`` correspondante.

    Raises:
        UnsupportedBrandError: Si la valeur ne correspond à aucune marque connue.
    """
    key = raw.strip().lower()

    if key in _BRAND_ALIASES:
        return _BRAND_ALIASES[key]

    for brand in Brand:
        if brand.value.lower() == key:
            return brand

    raise UnsupportedBrandError(
        f"Marque non reconnue : '{raw}'. "
        f"Valeurs supportées : {[b.value for b in Brand]}"
    )
