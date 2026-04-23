"""Test end-to-end du pipeline multimodal sur des donnees reelles BigQuery.

Recupere de vraies publicites depuis BigQuery, les traite avec le pipeline
LLM (Steps 1-4) et affiche des resultats structures et lisibles.

Utilisation :
    # Traiter 1 pub d'une marque (defaut: CHANEL, limit=1)
    python scripts/real_pipeline_check.py

    # Traiter une pub specifique par ID
    python scripts/real_pipeline_check.py --ad-id <platform_ad_id>

    # Traiter 3 pubs DIOR
    python scripts/real_pipeline_check.py --brand DIOR --limit 3

    # Mode texte seul (ignorer les medias pour aller plus vite)
    python scripts/real_pipeline_check.py --text-only
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Fix pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from google.cloud import bigquery

from ginjer_exercice.config import get_settings
from ginjer_exercice.data_access.bigquery_client import BigQueryClient
from ginjer_exercice.llm.factory import get_provider
from ginjer_exercice.observability.prompts import PromptRegistry
from ginjer_exercice.observability.tracing import pipeline_trace
from ginjer_exercice.pipeline import step1_universe, step2_products, step3_classify, step4_name
from ginjer_exercice.schemas.ad import Ad, Brand
from ginjer_exercice.schemas.products import FinalProductLabel
from ginjer_exercice.taxonomy.loader import load_taxonomy

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("real_pipeline_check")


# ── Helpers d'affichage ──────────────────────────────────────────

def separator(title: str, char: str = "=", width: int = 70) -> None:
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_ad_summary(ad: Ad, index: int) -> None:
    """Affiche un resume clair de la pub recuperee."""
    separator(f"PUB #{index} - {ad.platform_ad_id} [{ad.brand.value}]", "-")
    
    text_count = len(ad.texts)
    media_count = len(ad.media_urls)
    print(f"  Textes: {text_count} | Medias: {media_count}")
    
    for i, t in enumerate(ad.texts):
        label = f"  Texte[{i}]"
        if t.title:
            print(f"{label} Titre:   {t.title[:100]}")
        if t.body_text:
            preview = t.body_text[:120].replace("\n", " ")
            print(f"{label} Body:    {preview}{'...' if len(t.body_text) > 120 else ''}")
        if t.caption:
            print(f"{label} Caption: {t.caption[:100]}")
        if t.url:
            print(f"{label} URL:     {t.url[:80]}")
    
    if ad.media_urls:
        for i, url in enumerate(ad.media_urls[:5]):
            print(f"  Media[{i}]: {url[:90]}{'...' if len(url) > 90 else ''}")
        if len(ad.media_urls) > 5:
            print(f"  ... et {len(ad.media_urls) - 5} autre(s)")


def process_ad(
    ad: Ad,
    llm,
    registry: PromptRegistry,
    taxonomy,
    text_only: bool,
) -> dict:
    """Execute le pipeline Steps 1-4 sur une pub et retourne un resume."""
    
    result = {
        "ad_id": ad.platform_ad_id,
        "brand": ad.brand.value,
        "status": "ok",
        "universes": [],
        "products": [],
        "errors": [],
        "llm_calls": 0,
        "duration_ms": 0,
    }

    # Si text-only, on vide les media_urls pour eviter les appels GCS
    working_ad = ad
    if text_only and ad.media_urls:
        working_ad = Ad(
            platform_ad_id=ad.platform_ad_id,
            brand=ad.brand,
            texts=ad.texts,
            media_urls=[],
        )
        print("  [INFO] Mode text-only: medias ignores")
    
    start = time.monotonic()
    
    with pipeline_trace(working_ad, session_id="real_pipeline_check") as trace:
        
        # ── Step 1: Univers ───────────────────────────────
        print("\n  [STEP 1] Detection des univers...")
        try:
            u_result = step1_universe.execute(
                working_ad,
                llm_provider=llm,
                prompt_registry=registry,
                trace=trace,
            )
            result["llm_calls"] += 1
            result["universes"] = [
                {"name": u.universe, "confidence": u.confidence}
                for u in u_result.detected_universes
            ]
            for u in u_result.detected_universes:
                print(f"    -> {u.universe} (conf: {u.confidence:.2f}) - {u.reasoning[:60]}")
        except Exception as e:
            print(f"    [FAIL] {e}")
            result["errors"].append(f"Step1: {e}")
            result["status"] = "failed_step1"
            result["duration_ms"] = int((time.monotonic() - start) * 1000)
            return result

        # ── Step 2: Produits ──────────────────────────────
        print("\n  [STEP 2] Detection des produits...")
        try:
            products = step2_products.execute(
                working_ad,
                u_result,
                llm_provider=llm,
                prompt_registry=registry,
                trace=trace,
            )
            result["llm_calls"] += 1
            print(f"    -> {len(products)} produit(s) detecte(s)")
        except Exception as e:
            print(f"    [FAIL] {e}")
            result["errors"].append(f"Step2: {e}")
            result["status"] = "failed_step2"
            result["duration_ms"] = int((time.monotonic() - start) * 1000)
            return result

        # ── Steps 3-4 par produit ─────────────────────────
        for i, prod in enumerate(products):
            prod_entry = {
                "index": i + 1,
                "description": prod.raw_description[:80],
                "universe": prod.universe,
                "color": prod.color.value,
                "importance": prod.importance,
                "classification": None,
                "name": None,
                "valid_taxonomy": None,
                "errors": [],
            }

            print(f"\n    [PRODUIT {i+1}/{len(products)}] {prod.raw_description[:70]}...")
            print(f"      Univers: {prod.universe} | Couleur: {prod.color.value} | Importance: {prod.importance}/5")

            # Step 3: Classification
            print(f"      [STEP 3] Classification taxonomique...")
            c_result = None
            try:
                c_result = step3_classify.execute(
                    prod,
                    working_ad,
                    taxonomy,
                    llm_provider=llm,
                    prompt_registry=registry,
                    trace=trace,
                )
                result["llm_calls"] += 1
                prod_entry["classification"] = f"{c_result.universe} > {c_result.category} > {c_result.subcategory}"
                
                # Validation taxonomique
                is_valid = taxonomy.is_valid_path(c_result.universe, c_result.category, c_result.subcategory)
                is_terminal = taxonomy.is_terminal_category(c_result.universe, c_result.category)
                prod_entry["valid_taxonomy"] = is_valid or is_terminal
                
                status = "[OK]" if (is_valid or is_terminal) else "[WARN Invalid path]"
                print(f"        {status} {c_result.universe} > {c_result.category} > {c_result.subcategory} (conf: {c_result.confidence:.2f})")
            except Exception as e:
                print(f"        [FAIL] {e}")
                prod_entry["errors"].append(f"Step3: {e}")

            # Step 4: Nom
            print(f"      [STEP 4] Extraction du nom...")
            try:
                n_result = step4_name.execute(
                    prod,
                    c_result if c_result else None,
                    working_ad,
                    llm_provider=llm,
                    prompt_registry=registry,
                    trace=trace,
                )
                result["llm_calls"] += 1
                
                if n_result and n_result.name:
                    prod_entry["name"] = n_result.name
                    print(f"        [OK] '{n_result.name}' (source: {n_result.source}, conf: {n_result.confidence:.2f})")
                else:
                    print(f"        [--] Aucun nom explicite (fallback Phase 6 requis)")
            except Exception as e:
                print(f"        [FAIL] {e}")
                prod_entry["errors"].append(f"Step4: {e}")

            result["products"].append(prod_entry)

    result["duration_ms"] = int((time.monotonic() - start) * 1000)
    
    # Determiner le statut global
    has_errors = any(p["errors"] for p in result["products"])
    if has_errors:
        result["status"] = "partial"
    
    return result


def print_summary(results: list[dict], total_time_ms: int) -> None:
    """Affiche le tableau recapitulatif final."""
    separator("RECAPITULATIF", "=")
    
    # En-tete du tableau
    print(f"\n  {'Ad ID':<25} {'Brand':<12} {'Univers':<20} {'Produits':>8} {'Noms':>6} {'Taxo OK':>8} {'Statut':<10} {'Duree':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*20} {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    
    for r in results:
        ad_id = r["ad_id"][:24]
        brand = r["brand"][:11]
        universes = ", ".join(u["name"] for u in r["universes"][:3])[:19]
        n_products = len(r["products"])
        n_names = sum(1 for p in r["products"] if p.get("name"))
        n_taxo_ok = sum(1 for p in r["products"] if p.get("valid_taxonomy"))
        status = r["status"]
        duration = f"{r['duration_ms']}ms"
        
        print(f"  {ad_id:<25} {brand:<12} {universes:<20} {n_products:>8} {n_names:>6} {n_taxo_ok:>8} {status:<10} {duration:>8}")
    
    # Stats globales
    total_ads = len(results)
    total_products = sum(len(r["products"]) for r in results)
    total_llm_calls = sum(r["llm_calls"] for r in results)
    total_names = sum(1 for r in results for p in r["products"] if p.get("name"))
    total_valid = sum(1 for r in results for p in r["products"] if p.get("valid_taxonomy"))
    success = sum(1 for r in results if r["status"] == "ok")
    
    print(f"\n  --- Stats globales ---")
    print(f"  Pubs traitees:     {total_ads} ({success} OK, {total_ads - success} avec erreurs)")
    print(f"  Produits detectes: {total_products}")
    print(f"  Noms extraits:     {total_names}/{total_products}")
    print(f"  Taxonomie valide:  {total_valid}/{total_products}")
    print(f"  Appels LLM:        {total_llm_calls}")
    print(f"  Duree totale:      {total_time_ms}ms ({total_time_ms/1000:.1f}s)")
    
    if total_products > 0:
        print(f"  Temps moyen/pub:   {total_time_ms // total_ads}ms")
    
    # Detail des produits
    separator("DETAIL DES PRODUITS DETECTES", "-")
    for r in results:
        for p in r["products"]:
            name = p.get("name") or "(pas de nom)"
            classif = p.get("classification") or "(non classifie)"
            taxo = "[OK]" if p.get("valid_taxonomy") else "[??]"
            errors = f" | ERREURS: {', '.join(p['errors'])}" if p["errors"] else ""
            print(f"  {r['ad_id'][:20]} | P{p['index']} | {name:<25} | {classif:<40} {taxo}{errors}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test end-to-end du pipeline sur des donnees reelles BigQuery"
    )
    parser.add_argument("--ad-id", help="ID specifique d'une pub a traiter")
    parser.add_argument(
        "--brand",
        choices=[b.value for b in Brand],
        default="CHANEL",
        help="Marque a filtrer (defaut: CHANEL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Nombre de pubs a traiter (defaut: 1)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Ignorer les medias (traitement texte seul, plus rapide)",
    )
    args = parser.parse_args()

    separator("PIPELINE E2E - DONNEES REELLES BIGQUERY")
    
    # ── 1. Init infrastructure ────────────────────────────────
    print("\n[INIT] Configuration...")
    settings = get_settings()
    
    # Injecter les credentials GCP dans l'env pour BigQuery
    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    
    # BigQuery
    print("  BigQuery client...", end=" ")
    project_id = settings.gcp_project_id or "ginjer-440122"
    bq_raw = bigquery.Client(project=project_id)
    bq_client = BigQueryClient(bq_client=bq_raw)
    print(f"OK (projet: {project_id})")
    
    # LLM Provider
    print("  LLM provider...", end=" ")
    llm = get_provider(
        settings.llm_provider,
        use_vertex=True,
        project_id=settings.gcp_project_id,
    )
    print(f"OK ({llm.name})")
    
    # Prompts
    print("  Prompt registry...", end=" ")
    registry = PromptRegistry()
    print("OK")
    
    # ── 2. Fetch des pubs ─────────────────────────────────────
    separator("RECUPERATION DES PUBS DEPUIS BIGQUERY", "-")
    
    ads: list[Ad] = []
    
    if args.ad_id:
        print(f"  Recuperation de la pub: {args.ad_id}")
        try:
            ad = bq_client.fetch_ad(args.ad_id)
            ads = [ad]
            print(f"  [OK] Pub trouvee ({ad.brand.value})")
        except Exception as e:
            print(f"  [FAIL] {e}")
            sys.exit(1)
    else:
        brand = Brand(args.brand)
        print(f"  Recuperation de {args.limit} pub(s) pour {brand.value}...")
        try:
            ads = bq_client.fetch_ads_by_brand(brand, limit=args.limit)
            print(f"  [OK] {len(ads)} pub(s) recuperee(s)")
        except Exception as e:
            print(f"  [FAIL] {e}")
            sys.exit(1)
    
    if not ads:
        print("  [WARN] Aucune pub trouvee. Verifiez la marque ou l'ID.")
        sys.exit(1)
    
    # Afficher un resume de chaque pub
    for i, ad in enumerate(ads, 1):
        print_ad_summary(ad, i)
    
    # ── 3. Taxonomie ──────────────────────────────────────────
    print(f"\n[TAXO] Chargement de la taxonomie pour {ads[0].brand.value}...", end=" ")
    taxonomy = load_taxonomy(ads[0].brand)
    n_universes = len(taxonomy.get_universes())
    print(f"OK ({n_universes} univers)")
    
    # ── 4. Execution du pipeline ──────────────────────────────
    separator("EXECUTION DU PIPELINE (Steps 1-4)", "=")
    
    results: list[dict] = []
    total_start = time.monotonic()
    
    for i, ad in enumerate(ads, 1):
        separator(f"TRAITEMENT PUB {i}/{len(ads)} - {ad.platform_ad_id}", "-")
        
        # Recharger la taxonomie si la marque change
        if i > 1 and ad.brand != ads[i-2].brand:
            taxonomy = load_taxonomy(ad.brand)
        
        r = process_ad(ad, llm, registry, taxonomy, args.text_only)
        results.append(r)
    
    total_time = int((time.monotonic() - total_start) * 1000)
    
    # ── 5. Recap ──────────────────────────────────────────────
    print_summary(results, total_time)
    
    print(f"\n{'=' * 70}")
    print("  [DONE] Pipeline termine.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
