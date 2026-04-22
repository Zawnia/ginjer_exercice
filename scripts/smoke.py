"""Smoke test pour valider l'infrastructure d'observabilité Langfuse (Phase 4).

Utilisation:
    uv run python scripts/smoke.py
"""

import sys
import uuid
import time

from src.ginjer_exercice.config import get_settings
from src.ginjer_exercice.observability import (
    get_langfuse_client,
    pipeline_trace,
    step_span,
    PromptRegistry,
)
from src.ginjer_exercice.schemas.ad import Ad, Brand, AdText

def run_smoke_test():
    print("--- Démarrage du Smoke Test Phase 4 ---")
    
    # 1. Vérification configuration
    settings = get_settings()
    print(f"✅ Configuration chargée. Langfuse activé: {settings.langfuse_enabled}")
    if not settings.langfuse_enabled:
        print("Test arrêté car Langfuse est désactivé.")
        sys.exit(0)
        
    # 2. Client Langfuse
    langfuse = get_langfuse_client()
    if langfuse is None:
        print("❌ Échec d'initialisation du client Langfuse (clés manquantes ou erreur).")
        sys.exit(1)
        
    if not langfuse.auth_check():
        print("❌ Échec de l'authentification avec le serveur Langfuse.")
        sys.exit(1)
    print("✅ Authentification Langfuse réussie.")

    # 3. Création des mocks
    session_id = f"batch_smoke_{uuid.uuid4().hex[:6]}"
    ad_id = f"smoke_ad_{uuid.uuid4().hex[:6]}"
    
    mock_ad = Ad(
        platform_ad_id=ad_id,
        brand=Brand.CHANEL,
        texts=[AdText(title="Sac Classique", body_text="Découvrez la nouvelle collection printemps.")],
        media_urls=["http://example.com/image.jpg"]
    )

    # 4. Exécution tracée
    registry = PromptRegistry()
    
    print(f"Ouverture de la trace pour la pub {ad_id}...")
    with pipeline_trace(mock_ad, session_id=session_id) as trace_ctx:
        # Note: dans le SDK v4, start_as_current_observation renvoie un contexte
        # Les IDs (trace_id, observation_id) peuvent être accédés
        trace_id = trace_ctx.trace_id if hasattr(trace_ctx, 'trace_id') else None
        print(f"✅ Trace ouverte (ID: {trace_id})")
        
        # 5. Span enfant
        with step_span("step_1_universe", input_payload={"text": mock_ad.all_text()}):
            print("✅ Span 'step_1_universe' ouvert")
            
            # 6. Récupération Prompt
            prompt = registry.get("pipeline/universe")
            print(f"✅ Prompt 'pipeline/universe' récupéré (source: {prompt.source})")
            
            compiled_prompt = prompt.compile(brand=mock_ad.brand.value, ad_text=mock_ad.all_text())
            
            # 7. Mock Generation
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="llm_universe_call",
                input={"prompt": compiled_prompt},
                model=prompt.config.get("model", "mock-model"),
                model_parameters=prompt.config,
                prompt=prompt.name if prompt.source == "langfuse" else None
            ) as generation_ctx:
                
                # Simuler une latence
                time.sleep(0.5)
                
                mock_response = "Universe: Mode\nConfidence: 0.95"
                generation_ctx.update(output={"text": mock_response})
                print("✅ Génération mock enregistrée")
                
                generation_id = generation_ctx.id if hasattr(generation_ctx, 'id') else None
                
            # 8. Score
            print("Enregistrement du score...")
            # On log un faux score pour valider l'API
            langfuse.score(
                trace_id=trace_id,
                observation_id=generation_id,
                name="taxonomy_coherence",
                value=1.0,
                comment="Smoke test: mode > sacs"
            )
            print("✅ Score 'taxonomy_coherence' attaché")

    # 9. Flush
    print("Flush des données vers le serveur...")
    langfuse.flush()
    print("✅ Flush terminé.")
    
    print("\n🎉 Phase 4 smoke test passed ! 🎉")
    print(f"Allez voir la trace dans l'interface Langfuse (URL: {settings.langfuse_base_url})")
    if trace_id:
        print(f"Recherchez la trace ID: {trace_id}")

if __name__ == "__main__":
    run_smoke_test()
