"""Smoke test pour valider la couche data_access — BigQuery + Media.

Vérifie que :
- L'authentification ADC fonctionne
- La table BigQuery est accessible
- Le parsing des lignes et la normalisation des marques sont corrects
- Le téléchargement de médias fonctionne (optionnel)

Utilisation :
    uv run python scripts/smoke_data_access.py
    uv run python scripts/smoke_data_access.py --ad-id <id_specifique>
    uv run python scripts/smoke_data_access.py --skip-media
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Fix pythonpath for running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("smoke_data_access")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test data_access layer")
    parser.add_argument("--ad-id", help="ID spécifique d'une publicité à récupérer")
    parser.add_argument("--skip-media", action="store_true", help="Ne pas tester le téléchargement de médias")
    args = parser.parse_args()

    print("=" * 60)
    print("  SMOKE TEST — data_access layer")
    print("=" * 60)

    # ── 1. BigQuery Client ─────────────────────────────────────
    print("\n[INIT] Initialisation du client BigQuery...")
    try:
        import os
        from src.ginjer_exercice.config import get_settings
        settings = get_settings()
        
        # Inject the credential path into os.environ for google.cloud.bigquery
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

        from google.cloud import bigquery
        project_id = settings.gcp_project_id or "ginjer-440122"
        bq_raw = bigquery.Client(project=project_id)
        print(f"[OK] Client BigQuery initialisé (Projet: {project_id})")
    except Exception as exc:
        print(f"[FAIL] Échec de l'initialisation BigQuery : {exc}")
        print("   -> Avez-vous exécuté : gcloud auth application-default login ?")
        sys.exit(1)

    from src.ginjer_exercice.data_access.bigquery_client import BigQueryClient
    client = BigQueryClient(bq_client=bq_raw)

    # ── 2. count_ads() ─────────────────────────────────────────
    print("\n[COUNT] Comptage des publicités...")
    try:
        total = client.count_ads()
        print(f"[OK] {total} publicités dans la table")
        if total == 0:
            print("[WARN] La table est vide — vérifiez le dataset")
            sys.exit(1)
    except Exception as exc:
        print(f"[FAIL] count_ads() a échoué : {exc}")
        sys.exit(1)

    # ── 3. fetch_ad() ──────────────────────────────────────────
    if args.ad_id:
        ad_id = args.ad_id
        print(f"\n[FETCH] Récupération de la publicité : {ad_id}")
    else:
        # Fetch first available ad
        print("\n[FETCH] Récupération de la première publicité disponible...")
        from src.ginjer_exercice.schemas.ad import Brand
        for brand in Brand:
            ads = client.fetch_ads_by_brand(brand, limit=1)
            if ads:
                ad_id = ads[0].platform_ad_id
                break
        else:
            print("[FAIL] Aucune publicité trouvée pour aucune marque")
            sys.exit(1)

    try:
        start = time.monotonic()
        ad = client.fetch_ad(ad_id)
        latency = (time.monotonic() - start) * 1000
        print(f"[OK] Publicité récupérée en {latency:.0f}ms")
        print(f"   ID:     {ad.platform_ad_id}")
        print(f"   Brand:  {ad.brand.value}")
        print(f"   Texts:  {len(ad.texts)} entrée(s)")
        print(f"   Media:  {len(ad.media_urls)} URL(s)")

        if ad.texts:
            first_text = ad.texts[0]
            print(f"   Titre:  {first_text.title or '(aucun)'}")
            body_preview = (first_text.body_text or "")[:80]
            print(f"   Body:   {body_preview}{'...' if len(first_text.body_text or '') > 80 else ''}")

        if ad.media_urls:
            for i, url in enumerate(ad.media_urls[:3]):
                print(f"   URL[{i}]: {url[:80]}...")
    except Exception as exc:
        print(f"[FAIL] fetch_ad('{ad_id}') a échoué : {exc}")
        sys.exit(1)

    # ── 4. Media download (optionnel) ──────────────────────────
    if not args.skip_media and ad.media_urls:
        print(f"\n[DOWNLOAD] Test de téléchargement de médias ({len(ad.media_urls)} URL(s))...")
        try:
            import httpx
            from src.ginjer_exercice.data_access.media_fetcher import MediaFetcher

            http_client = httpx.Client(timeout=30.0, follow_redirects=True)
            fetcher = MediaFetcher(client=http_client)

            first_url = ad.media_urls[0]
            print(f"   Téléchargement : {first_url[:80]}...")

            start = time.monotonic()
            media = fetcher.download(first_url)
            latency = (time.monotonic() - start) * 1000

            print(f"[OK] Média téléchargé en {latency:.0f}ms")
            print(f"   Kind:  {media.kind.value}")
            print(f"   MIME:  {media.mime_type}")
            print(f"   Size:  {media.size_bytes:,} octets")
        except Exception as exc:
            print(f"[WARN] Téléchargement échoué (non bloquant) : {exc}")
    elif args.skip_media:
        print("\n[SKIP] Téléchargement de médias ignoré (--skip-media)")
    else:
        print("\n[SKIP] Aucune URL de média à tester")

    # ── 5. Résumé ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [SUCCESS] SMOKE TEST PASSED !")
    print("=" * 60)
    print("\nLa couche data_access est fonctionnelle.")
    print("Prochaine étape : exécuter le pipeline complet sur cette publicité.")


if __name__ == "__main__":
    main()
