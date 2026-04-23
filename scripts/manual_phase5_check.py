"""Script de vérification manuelle pour la Phase 5 (Steps 1-4)."""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ajout du dossier src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ginjer_exercice.schemas.ad import Ad, Brand, AdText
from ginjer_exercice.config import get_settings
from ginjer_exercice.llm.factory import get_provider
from ginjer_exercice.observability.prompts import PromptRegistry
from ginjer_exercice.taxonomy.loader import load_taxonomy
from ginjer_exercice.observability.tracing import pipeline_trace
from ginjer_exercice.pipeline import step1_universe, step2_products, step3_classify, step4_name
from ginjer_exercice.observability.client import get_langfuse_client


def main():
    """Point d'entrée pour tester les étapes 1 à 4 du pipeline."""
    print("=" * 50)
    print("=== MANUAL PHASE 5 CHECK ===")
    print("=" * 50)
    
    ad = Ad(
        platform_ad_id="manual_check_001",
        brand=Brand.CHANEL,
        texts=[
            AdText(
                title="N°5 L'EAU",
                body_text="Découvrez la nouvelle fragrance CHANEL N°5 L'EAU. Un parfum floral et frais pour le printemps. Également, explorez notre nouvelle collection de sacs classiques matelassés en cuir noir.",
                url="https://chanel.com/n5"
            )
        ],
        media_urls=[]
    )
    print(f"Fixture prête: Ad ID '{ad.platform_ad_id}' | Marque '{ad.brand.value}'")
    
    print("\nInitialisation des dépendances (Settings, LLM, Prompts, Taxonomy, Langfuse)...")
    settings = get_settings()
    
    # Résolution du problème d'API : Force l'utilisation du mode Vertex (Service Account)
    llm = get_provider(
        settings.llm_provider,
        use_vertex=True,
        project_id=settings.gcp_project_id
    )
    
    # Utilisation du nom exact de ta fonction : get_langfuse_client
    langfuse_client = get_langfuse_client()
    registry = PromptRegistry()
    taxonomy = load_taxonomy(ad.brand)
    
    print("\nLancement du pipeline avec trace Langfuse...")
    
    with pipeline_trace(ad, session_id="manual_check_session") as trace:
        
        print("\n--- STEP 1: Détection des univers ---")
        try:
            u_result = step1_universe.execute(
                ad, 
                llm_provider=llm, 
                prompt_registry=registry, 
                trace=trace
            )
            print("[OK] Détection terminée. Univers trouvés :")
            for u in u_result.detected_universes:
                print(f"  - {u.universe} (confiance: {u.confidence:.2f})")
        except Exception as e:
            print(f"[FAIL] Step 1 a échoué: {e}")
            return
            
        print("\n--- STEP 2: Détection des produits ---")
        try:
            p_result = step2_products.execute(
                ad, 
                universe_result=u_result, 
                llm_provider=llm, 
                prompt_registry=registry, 
                trace=trace
            )
            print(f"[OK] Détection terminée. Nombre de produits : {len(p_result)}")
        except Exception as e:
            print(f"[FAIL] Step 2 a échoué: {e}")
            return
            
        for i, prod in enumerate(p_result):
            print(f"\n[*] Produit #{i+1}:")
            print(f"  - Description brute : {prod.raw_description}")
            print(f"  - Univers retenu  : {prod.universe}")
            print(f"  - Importance      : {prod.importance}/5")
            
            color_val = prod.color.value if hasattr(prod.color, 'value') else prod.color
            print(f"  - Couleur         : {color_val}")
            
            print("\n  --- STEP 3: Classification ---")
            try:
                c_result = step3_classify.execute(
                    prod, 
                    ad, 
                    taxonomy=taxonomy, 
                    llm_provider=llm, 
                    prompt_registry=registry, 
                    trace=trace
                )
                print(f"  [OK] Classifié: {c_result.universe} > {c_result.category} > {c_result.subcategory} (confiance: {c_result.confidence:.2f})")
                
                product_type = getattr(c_result, "product_type", None)
                is_valid = taxonomy.is_valid_path(c_result.universe, c_result.category, c_result.subcategory)
                is_terminal = taxonomy.is_terminal_category(c_result.universe, c_result.category)
                
                if is_valid or is_terminal:
                    print(f"  [OK] Validation métier : Chemin valide dans la taxonomie {ad.brand.value}.")
                else:
                    print(f"  [FAIL] Erreur métier : Le chemin retourné N'EST PAS valide dans la taxonomie.")
                    
            except Exception as e:
                print(f"  [FAIL] Step 3 a échoué: {e}")
                c_result = None
                
            print("\n  --- STEP 4: Extraction explicite du nom ---")
            if c_result is None:
                print("  [WARN] Step 4 ignoré car Step 3 a échoué.")
                continue
                
            try:
                n_result = step4_name.execute(
                    prod, 
                    classification=c_result, 
                    ad=ad, 
                    llm_provider=llm, 
                    prompt_registry=registry, 
                    trace=trace
                )
                if n_result is None or n_result.name is None:
                    print("  [INFO] Aucun nom explicite trouvé (Normal : déclenchera le fallback Phase 6).")
                else:
                    print(f"  [OK] Nom trouvé : '{n_result.name}'")
                    print(f"     (source: {n_result.source}, confiance: {n_result.confidence:.2f})")
            except Exception as e:
                print(f"  [FAIL] Step 4 a échoué: {e}")

    print("\n" + "=" * 60)
    print("=== GRILLE DE VALIDATION MANUELLE ===")
    print("[ ] Step 1 : Univers corrects ? (Fragrance, Fashion/Bags)")
    print("[ ] Step 2 : Nombre de produits plausible ?")
    print("[ ] Step 3 : Statut [OK] (chemin valide) ?")
    print("[ ] Step 4 : 'N°5 L'EAU' trouvé et sac 'None' ?")
    print("[ ] Langfuse : Trace visible sur l'interface locale ?")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()