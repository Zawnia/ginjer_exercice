# ROADMAP - Projet Ginjer Exercice

**Etat au 24 avril 2026**

## 1. Vue d'ensemble

Pipeline de detection et de labellisation de produits dans des publicites de marques de luxe.

Marques ciblees :
- Chanel
- Dior
- Louis Vuitton
- Balenciaga
- Maison Francis Kurkdjian

Sources et composants principaux :
- donnees ads : BigQuery `ginjer-440122.ia_eng_interview.ads`
- medias : URLs publiques, principalement GCS
- persistance de sortie : SQLite
- observabilite : Langfuse SDK v4
- provider LLM principal : Gemini

Architecture en couches :
- `schemas/`
- `pipeline/`
- `data_access/`
- `llm/`
- `observability/`
- `taxonomy/`
- `web_search/` (reserve pour la suite)

Regle d'or :
- aucun SDK externe ne doit fuir hors de sa couche dediee

Stack technique actuelle :
- Python 3.11+
- Pydantic v2
- `google-genai`
- `httpx`
- SQLite
- Langfuse SDK v4

## 2. Etat reel du projet

### 2.1 Termine et valide en execution

- Schemas Pydantic utilises par le pipeline et la persistance.
- Taxonomies :
  - Dior et MFK chargees depuis fichiers
  - Chanel derivee du CSV `product_categorisation`
  - Louis Vuitton et Balenciaga generees via LLM
  - flow `refresh-taxonomy` operationnel
- Data access :
  - `bigquery_client.py` : fetch unitaire et fetch par marque operationnels
  - `media_fetcher.py` : telechargement media operationnel
  - `results_repository.py` : persistance SQLite active dans le flow produit
- Abstraction LLM :
  - `LLMProvider` generique
  - `gemini_provider.py` valide en execution reelle
  - support de messages ordonnes multi-parts (`text`, `media`)
  - structured outputs Pydantic fonctionnels
- Observabilite :
  - `TraceContext` et spans pipeline operationnels
  - traces, generations, usage et scores publies en best-effort
  - warnings runtime collectes et rattaches au `PipelineOutput`
- Steps pipeline 1 a 4 implementes :
  - `step1_universe`
  - `step2_products`
  - `step3_classify`
  - `step4_name`
- Correction prompts :
  - tous les `prompts/*.yaml` sont alignes a `max_tokens: 4000`
- Multimodal P0 :
  - `step2` telecharge les medias via `MediaFetcher`
  - seules les images sont injectees au LLM
  - les videos sont ignorees explicitement en P0
  - fallback texte seul si aucune image exploitable n'est disponible
- Orchestrateur minimal :
  - `pipeline/orchestrator.py`
  - enchaine `step1 -> step2 -> boucle(step3 -> step4) -> scoring -> persist`
- CLI produit :
  - commande `refresh-taxonomy`
  - commande `process-ad`
  - options globales `--verbose` et `--debug`
- Documentation livree :
  - `README.md`
  - `docs/USAGE.md`
  - `docs/DEPLOYMENT.md`
- Validation E2E reelle disponible :
  - `scripts/validate_process_ad_real_e2e.py`
  - preuve sur BigQuery reel + media reel + LLM reel + orchestrateur reel + CLI reelle + SQLite reelle

### 2.2 Partiellement fait

- Multimodal :
  - seul `step2` consomme des images telechargees
  - `step1`, `step3` et `step4` restent text-only
  - le support video n'est pas livre dans le flow produit
- Scoring :
  - `taxonomy_coherence` livre
  - `confidence` agregee livree
  - `llm_judge` non implemente
- OpenAI provider :
  - support du nouveau contrat multi-parts
  - non valide en E2E reel
  - pas de support video assume
- Observabilite Langfuse :
  - le besoin produit est couvert
  - certains details restent pragmatiques et non completement harmonises avec la vision historique de la couche observability
- Documentation deployment :
  - le document est present
  - l'infra applicative complete n'est pas encore livree

### 2.3 Non commence ou hors P0

- `step5_fallback.py`
  - fallback catalogue / web non implemente
- `web_search/`
  - abstraction reservee pour `step5`
- `process-batch`
  - absent
- scoring `LLM-as-judge`
  - absent
- Dockerfile applicatif
  - absent
- `docker-compose.yml` Langfuse self-host
  - absent

## 3. Validation executee a ce stade

### 3.1 Validation logicielle

- tests unitaires sur :
  - prompts
  - contrat LLM multi-parts
  - `step2` multimodal
  - orchestrateur
  - CLI
  - repository
- `compileall` / `py_compile` executes sur les fichiers critiques modifies

### 3.2 Validation end-to-end

Deux niveaux de validation existent :

1. `scripts/e2e_process_ad_cli_multimodal.py`
- smoke test d'integration locale
- utile pour verifier le cablage, pas comme preuve forte terrain

2. `scripts/validate_process_ad_real_e2e.py`
- validation reelle
- BigQuery reel
- media image reel
- provider LLM reel
- orchestrateur reel
- CLI lancee en subprocess
- persistance SQLite verifiee

Cas reel deja valide :
- marque : `CHANEL`
- `ad_id=1000142752321014`

Ce script prouve explicitement :
- que `step2` envoie bien une image binaire au LLM
- que l'orchestrateur persiste
- que la CLI produit un run complet

## 4. Dette technique et zones fragiles

### 4.1 Instabilite Gemini sur sorties structurees

Probleme observe en execution reelle :
- Gemini renvoie parfois un `response.text` JSON tronque
- erreurs typiques : `EOF while parsing a string` / `EOF while parsing a value`
- touche au moins `UniverseResult` et `ProductClassification`

Etat actuel :
- mitigation de reparation JSON en place dans `gemini_provider.py`
- warning runtime explicite remonte dans `PipelineOutput.warnings`
- logs diagnostiques enrichis

Lecture rigoureuse :
- le pipeline est plus resilient
- mais la cause racine n'est pas resolue
- un run avec reparation ne doit pas etre interprete comme parfaitement sain

Reference :
- `.memory/gemini_structured_output_instability_2026-04-23.md`

### 4.2 Variance LLM inter-runs

Constat :
- deux runs reels successifs sur une meme pub peuvent produire
  - des univers differents
  - des chemins taxonomiques differents
  - des scores de confiance differents

Impact :
- acceptable pour une P0 demonstrable
- insuffisant pour un niveau production stricte si l'on attend une forte reproductibilite

### 4.3 Multimodal encore partiel

- `step2` seulement
- videos ignorees
- chemins media hors `http/https` non traites par le flux `MediaFetcher` P0

### 4.4 Observability encore pragmatique

- les besoins d'exploitation sont couverts
- certaines conventions restent heterogenes :
  - `usage_details` non totalement alignes avec le contrat Langfuse canonique
  - publication de scores en best-effort seulement

### 4.5 Surface produit encore minimale

- mono-ad uniquement
- absence de fallback `step5`
- absence de reprise sur erreur metier partielle

## 5. Backlog priorise

### 5.1 Principe de priorisation

Apres livraison de la P0, la priorite n'est plus de cabler le chemin critique, mais de :
- stabiliser le comportement reel
- rendre les runs plus auditables
- reduire les zones de non-determinisme et les pansements techniques

### 5.2 P0 - Livree

La P0 de la section 4.2 initiale est consideree comme livree :

1. Correction `max_tokens`
- terminee

2. Multimodal `step2` images only
- termine

3. Orchestrateur minimal
- termine

4. CLI `process-ad`
- terminee

Documentation en parallele :
- `README.md`
- `docs/USAGE.md`
- `docs/DEPLOYMENT.md`
- livree

### 5.3 P0.5 - Stabilisation immediate

1. Diagnostic racine du probleme Gemini structured output
- mesurer finement les cas d'echec
- comparer `response.parsed` vs `response.text`
- comparer mode `response_schema` vs JSON texte brut
- definir une politique produit claire pour les runs "repares"

2. Instrumentation qualite
- consolider le suivi des `warnings`
- exposer clairement les runs degrades dans les validations et les docs
- eventuellement faire evoluer `needs_review` ou un flag qualite dedie

3. Validation repetee sur corpus fixe
- rejouer un petit corpus d'ads reelles
- mesurer stabilite, taux de warnings, variance taxonomique

### 5.4 P1 - Important

1. `step5` fallback
- `5a` catalogue enrichi par LLM
- `5b` verification web si le temps le permet

2. `process-batch`
- traitement multi-ads

3. Extension multimodale
- etendre le visuel a d'autres steps si le gain est demontre

4. Durcissement observability
- harmoniser `usage_details`
- clarifier la semantique des warnings et scores

5. OpenAI provider
- soit validation reelle et documentation claire
- soit declassement explicite comme scaffolding

### 5.5 P2 - Bonus

- scoring `LLM-as-judge`
- support video si utile et justifie
- infra complete :
  - Dockerfile
  - `docker-compose.yml`
  - packaging deploiement plus propre

## 6. Documentation et contrat produit

### 6.1 Point d'entree canonique

Le point d'entree produit P0 est :
- `ginjer_exercice.cli process-ad <ad_id>`

### 6.2 Limitations assumees

- pas de `step5`
- pas de `process-batch`
- pas de `llm_judge`
- multimodal partiel
- OpenAI non valide en E2E reel
- pipeline sensible a la variance du provider Gemini

### 6.3 Ce qui est considere comme vrai a date

- le pipeline tient en bout en bout sur un cas reel
- la persistance SQLite est branchee
- la CLI n'est plus un stub
- la documentation de base existe
- la stabilite qualitative n'est pas encore au niveau d'un produit totalement industrialise

## 7. Historique des phases

### 7.1 Phases completees

- **2026-04-22** - Phase 0 : bootstrap et configuration initiale
- **2026-04-22** - Phase 1 : schemas Pydantic
- **2026-04-22** - Phase 2 : abstraction `LLMProvider` + Gemini
- **2026-04-22** - Phase 3 : taxonomies
- **2026-04-22** - Phase 4 : wiring initial observability
- **2026-04-23** - Phase 5 : steps 1 a 4
- **2026-04-23** - Phase 5.5 : data access utilisable
- **2026-04-23 / 2026-04-24** - Phase 6 : livraison P0
  - prompts `max_tokens`
  - multimodal `step2`
  - orchestrateur
  - CLI `process-ad`
  - docs
  - validations E2E

### 7.2 Phase actuelle

- **Phase 6.5** - Stabilisation post-P0
  - investigation structured output Gemini
  - normalisation des signaux qualite
  - durcissement production

### 7.3 Phases a venir

- **Phase 7** - fallback `step5` + batch + durcissement
- **Phase 8** - scoring avance, infra complete, polish final

## 8. References

- source de verite historique :
  - `.memory/audit_2026-04-23.md`
  - `.memory/data_access.md`
  - `.memory/phase3_implementation.md`
  - `.memory/phase4_observability.md`
- cartographie P0 :
  - `.memory/p0_implementation_map_2026-04-23.md`
- note technique sur l'instabilite Gemini :
  - `.memory/gemini_structured_output_instability_2026-04-23.md`
