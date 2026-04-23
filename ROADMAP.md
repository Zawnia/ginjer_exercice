# ROADMAP — Projet Ginjer Exercice

**État au 23 avril 2026**

## 1. Vue d'ensemble

Pipeline multimodal de détection de produits dans des publicités de marques de luxe : Chanel, Dior, Louis Vuitton, Balenciaga, Maison Francis Kurkdjian.

- **Source de données** : BigQuery `ginjer-440122.ia_eng_interview.ads`
- **Médias** : URLs publiques GCS
- **Architecture** : couches strictement séparées
  - `schemas/`
  - `pipeline/`
  - `data_access/`
  - `llm/`
  - `observability/`
  - `web_search/`

**Règle d'or** : aucun SDK externe ne fuit hors de sa couche dédiée.

**Stack technique** :

- Python 3.11+
- Pydantic v2
- `google-genai` (provider par défaut, choisi pour le multimodal natif images + vidéo)
- Langfuse SDK v4
- `httpx`
- SQLite
- Docker Compose

## 2. État réel du projet

### 2.1 Terminé et validé en exécution

- Schémas Pydantic complets et utilisés par tous les consommateurs.
- Taxonomies :
  - Dior et MFK chargées depuis fichiers
  - Chanel dérivée du CSV `product_categorisation`
  - Louis Vuitton et Balenciaga générées via LLM
  - Flow `refresh-taxonomy` opérationnel
- Data access partiel :
  - `bigquery_client.py` : fetch ads OK, validé sur l’ad `1000142752321014`
  - `media_fetcher.py` : opérationnel
- Abstraction LLM :
  - `LLMProvider` (`base.py`)
  - `gemini_provider.py` (Vertex AI, validé en production)
  - factory model-agnostic
  - structured outputs Pydantic fonctionnels
- Observabilité Langfuse :
  - wrapper `TraceContext` conforme à la règle d’or
  - aucun SDK Langfuse hors de `observability/`
  - traces, spans nestés, generations avec usage tokens et coûts
  - 6 tests unitaires + 30 tests de steps passants
- Steps pipeline 1 à 4 implémentés et validés end-to-end en mode text-only :
  - **step1** : détection d’univers multi-univers
  - **step2** : détection de produits (`DetectedProduct`)
  - **step3** : classification taxonomique hiérarchique
  - **step4** : extraction de nom explicite
- Retry sur `ValidationError` fonctionnel dans `step3`.
- Prompts Langfuse versionnés dans Langfuse Prompt Management, avec fallback YAML local.
- Run E2E validé sur l’ad Chanel `1000142752321014` :
  - 3 produits détectés
  - 3 noms extraits
  - 3 taxonomies valides
  - 8 appels LLM tracés dans Langfuse avec coûts

### 2.2 Partiellement fait

- **Multimodal** :
  - `gemini_provider.py` a été choisi pour le multimodal natif
  - le chemin images/vidéo n’est pas encore câblé dans les steps
  - `real_pipeline_check.py` tourne actuellement avec `--text-only`
  - `LLMMessage` doit être étendu pour supporter des content parts mixtes
  - c’est le gap central du projet
- **`results_repository`** :
  - code SQLite présent dans `data_access/`
  - non branché au flow principal
  - aucune persistance lors du run E2E
- **Scoring** :
  - `taxonomy_coherence` est fait
  - il manque la confidence agrégée et le `LLM-as-judge`
  - `observability/scoring.py` doit être audité pour vérifier sa cohérence avec l’état actuel du code
- **CLI** :
  - `ginjer-exercice` est encore un stub affichant `Hello from ginjer-exercice!`
  - seul `refresh-taxonomy` est câblé
  - `process-ad` et `process-batch` manquent
- **Tests d’intégration** :
  - tests unitaires OK
  - il manque un vrai test d’intégration E2E avec provider LLM fake

### 2.3 Non commencé

- **Orchestrator** (`pipeline/orchestrator.py`) :
  - pas de flow principal intégré enchaînant  
    `step1 -> step2 -> boucle {step3 -> step4 -> step5} -> scoring -> persist`
  - aujourd’hui, `real_pipeline_check.py` joue le rôle de glue script
- **Step 5 fallback** (`pipeline/step5_fallback.py`) :
  - prévu pour les produits sans nom explicite
  - `5a` : enrichissement LLM avec catalogue
  - `5b` : vérification web
  - ni les sous-étapes ni le routage ne sont implémentés
- **Couche `web_search/`** :
  - abstraction `WebSearchProvider` prévue pour `5b`
  - non commencée
- **Scoring `LLM-as-judge`** :
  - prévu au design
  - non implémenté
- **Documentation** :
  - README fonctionnel
  - mais `docs/DEPLOYMENT.md`, `docs/USAGE.md` et les screenshots Langfuse restent à produire
- **Infra** :
  - `Dockerfile` applicatif
  - `docker-compose.yml` Langfuse self-host
  - `.env.example`

## 3. Dette technique et zones fragiles

- **Bug 2 latent : hallucination de catégorie par le LLM**
  - `step3` a réussi sur ce run grâce au retry
  - la cause racine n’est pas traitée
  - la correction 3 (`max_tokens -> 4000` dans tous les YAML) n’est pas encore appliquée
  - à faire avant livraison pour stabiliser le comportement
- **`openai_provider.py`**
  - présent mais jamais exercé en E2E
  - à documenter explicitement comme scaffolding, ou à retirer si hors scope
- **`usage_details`**
  - clés actuelles : `input_tokens` / `output_tokens`
  - non canoniques par rapport à Langfuse v4 (`input` / `output` / `total`)
  - les coûts remontent malgré tout, mais le mapping n’est pas idéal
- **`observability/scoring.py`**
  - signalé par un audit précédent
  - non revérifié manuellement à ce stade
- **Imports hétérogènes**
  - mélange possible entre `ginjer_exercice...` et `src.ginjer_exercice...`
  - signalé par audit, à vérifier

## 4. Backlog priorisé

### 4.1 Principe de priorisation

L’objectif est de livrer un pipeline complet, fonctionnel et démontrable, quitte à garder certains axes volontairement partiels, à condition qu’ils soient explicitement documentés comme tels.

### 4.2 P0 — Critique pour livraison

1. **Correction 3 — Bug 2 (`max_tokens`)**
   - passer `max_tokens` à `4000` dans tous les prompts YAML
   - quick win, faible coût, impact immédiat sur la stabilité

2. **Multimodal**
   - câbler le chemin image dans `step2` (détection produits), qui est le step bénéficiant le plus du visuel
   - étendre `LLMMessage` pour supporter des content parts mixtes
   - utiliser `media_fetcher` pour charger l’image
   - injecter l’image dans le prompt
   - tester sur l’ad Chanel déjà disponible avec image
   - c’est le sujet central de l’exercice

3. **Orchestrator minimal**
   - créer `pipeline/orchestrator.py`
   - enchaîner les steps
   - appeler `results_repository` pour la persistance
   - remplacer le script de vérification par une vraie entrée de production

4. **CLI `process-ad`**
   - prendre un `ad_id`
   - appeler l’orchestrator
   - persister les résultats
   - retourner un résumé d’exécution

### 4.3 P1 — Important si le temps le permet

1. **Step 5 fallback**
   - implémenter `5a` : LLM enrichi avec catalogue
   - implémenter `5b` : vérification web si le temps le permet
   - scope minimal acceptable : `5a` seul
   - documenter clairement que `5b` est prévu architecturalement mais non implémenté le cas échéant

2. **Scoring**
   - ajouter une confidence agrégée
   - auditer et corriger `observability/scoring.py`
   - `taxonomy_coherence` est déjà disponible

3. **CLI `process-batch`**
   - traitement de plusieurs ads

4. **Test d’intégration end-to-end**
   - avec provider LLM fake

### 4.4 P2 — Bonus si la livraison est stabilisée

- Implémenter le scoring `LLM-as-judge`
- Étendre le multimodal à d’autres steps :
  - `step1` pour l’univers
  - `step4` si pertinent
- Corriger les clés `usage_details` pour conformité Langfuse v4
- Finaliser ou retirer `openai_provider.py`

### 4.5 Documentation (en parallèle, de manière incrémentale)

- README d’usage avec commandes et exemples
- `docs/DEPLOYMENT.md` avec Langfuse self-host
- `docs/USAGE.md` avec flow pipeline + scoring
- Section **Architecture** du README :
  - schéma des couches
  - rappel de la règle d’or
- Section **Limitations & Extensions** :
  - multimodal partiel
  - `step5` fallback partiel
  - `LLM-as-judge` non implémenté
  - autres limites assumées
- Screenshots Langfuse :
  - traces nestées
  - generations avec coûts
  - prompts versionnés

## 5. Historique des phases

### 5.1 Phases complétées

- **2026-04-22** — Phase 0 : Bootstrap, Docker, config
- **2026-04-22** — Phase 1 : Schémas Pydantic complets
- **2026-04-22** — Phase 2 : Abstraction `LLMProvider` + implémentation Gemini
- **2026-04-22** — Phase 3 : Taxonomies
  - Dior
  - MFK
  - Chanel dérivée
  - LV / Balenciaga générées
- **2026-04-22** — Phase 4 : wiring initial observability
  - client Langfuse
  - `PromptRegistry`
  - context managers
- **2026-04-23** — Phase 5 : Steps 1 à 4 implémentés avec tests unitaires
- **2026-04-23** — Data access partiel :
  - BigQuery client
  - media fetcher
  - `results_repository` non branché

### 5.2 Phases en cours ou à venir

- **Phase 5.5** — Fix observabilité Langfuse (en finalisation)
  - commit 1 : `TraceContext` OK
  - commit 2 : `log_generation` aligné contrat legacy OK
  - commit 3 : `max_tokens` en cours
- **Phase 6** — Orchestrator + Multimodal + CLI
- **Phase 7** — Step 5 fallback + scoring complet
- **Phase 8** — Docs, infra, polish

## 6. Références

- Rapport d’audit du `2026-04-23` : `.memory/audit_2026-04-23.md`
- Historique mémoire :
  - `.memory/phase3_implementation.md`
  - `.memory/phase4_observability.md`
  - `.memory/data_access.md`