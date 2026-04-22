# Roadmap — Multimodal Product Detection Pipeline

> Document de référence du projet. À garder à la racine du repo et à fournir comme contexte à tout assistant de code (Claude Code, Cursor, etc.).

---

## Table des matières

1. [Contexte général](#1-contexte-général)
2. [Objectifs & livrables](#2-objectifs--livrables)
3. [Principes directeurs](#3-principes-directeurs)
4. [Architecture cible](#4-architecture-cible)
5. [Schémas de données](#5-schémas-de-données)
6. [Couche LLM abstraite](#6-couche-llm-abstraite)
7. [Taxonomies](#7-taxonomies)
8. [Le pipeline — étape par étape](#8-le-pipeline--étape-par-étape)
9. [Fallback de nommage](#9-fallback-de-nommage)
10. [Scoring & observabilité Langfuse](#10-scoring--observabilité-langfuse)
11. [Roadmap d'implémentation par phases](#11-roadmap-dimplémentation-par-phases)
12. [Règles de travail (assistant de code)](#12-règles-de-travail-assistant-de-code)
13. [Questions ouvertes à clarifier](#13-questions-ouvertes-à-clarifier)

---

## 1. Contexte général

Le projet est un test technique d'AI Engineer. Il s'agit de construire un **pipeline multimodal de détection de produits dans des publicités de marques de luxe**, avec une observabilité complète via Langfuse.

**Marques supportées** : Chanel, Dior, Louis Vuitton, Balenciaga, Maison Francis Kurkdjian (MFK).

**Source des données** : une table BigQuery `ginjer-440122.ia_eng_interview.ads` contenant des pubs avec textes et médias (images + vidéos via URLs publiques).

**Sortie attendue par pub** : une liste de produits identifiés, chacun avec :
- Universe (Fashion, Beauty, Home, Insurance — à confirmer, voir §13)
- Category → Subcategory → Product type (hiérarchique)
- Nom du produit (depuis le texte, ou via fallback enrichi + vérif web)
- Couleur
- Importance visuelle (0-5)

**Défis principaux** :
1. **Classification hiérarchique stricte** — chaque niveau contraint le suivant ; la taxonomie dépend de la marque.
2. **Identification du nom avec fallback intelligent** — si le nom n'est pas explicite, enrichissement LLM avec catalogue + vérification web, et flag `needs_review` si confiance faible.
3. **Traçabilité complète** — chaque étape, chaque appel LLM, chaque prompt versionné, tracés dans Langfuse.

---

## 2. Objectifs & livrables

### Livrables attendus

| # | Livrable | Description |
|---|----------|-------------|
| 1 | **Repo Git** | Code complet, structuré, testé |
| 2 | **`docs/DEPLOYMENT.md`** | Guide pas-à-pas : Docker, Langfuse, GCP, clés LLM |
| 3 | **`docs/USAGE.md`** | Doc de chaque fonction publique avec exemples call/response |
| 4 | **`Dockerfile` + `docker-compose.yml`** | Infra complète pour tourner en local |
| 5 | **Fonctions d'accès données** | BigQuery (ads) + médias (public URLs) + stockage résultats |
| 6 | **Screenshots Langfuse** | Traces, prompts versionnés, scores |
| 7 | **`README.md`** | Approche, choix d'archi, justification modèles, fallback, taxonomies, Langfuse, limites, **temps passé honnête** |

### Critères d'évaluation clés

- Qualité de l'observabilité Langfuse (c'est le cœur de l'évaluation).
- Respect des contraintes : model-agnostic, multimodal, taxonomies dynamiques, outputs structurés.
- Clarté de la documentation et de l'architecture.
- Pertinence des questions posées en amont.

---

## 3. Principes directeurs

### 3.1 Séparation des couches stricte

De l'intérieur vers l'extérieur :

1. **Schemas (Pydantic)** — contrats de données. Tout ce qui circule entre couches est typé.
2. **Domain logic (`pipeline/`)** — logique métier pure. Ne connaît ni BigQuery, ni Gemini, ni Langfuse directement.
3. **Adapters (`data_access/`, `llm/`, `observability/`, `web_search/`)** — traduisent le monde extérieur vers des types du domaine.
4. **Orchestration (`pipeline/orchestrator.py`)** — assemble les étapes, gère les erreurs.
5. **Entrypoint (`cli.py` / `main.py`)** — lance tout.

**Règle d'or** : si je veux remplacer Gemini par Claude, je touche UN fichier. Si BigQuery → Postgres, UN fichier. Si Langfuse change d'API, UN fichier.

### 3.2 Model-agnostic (concrètement)

Le pipeline ne fait JAMAIS `from google.genai import ...` ni `import openai`. Il fait toujours :

```python
from llm.base import LLMProvider
provider: LLMProvider = get_provider(settings.llm_provider)
result = provider.generate_structured(prompt, media, schema)
```

L'interface `LLMProvider` est abstraite (ABC ou Protocol). Les implémentations (Gemini, OpenAI, Anthropic) vivent dans des fichiers séparés, sélectionnées par un factory qui lit la config.

### 3.3 Multimodal = un seul appel par étape

Chaque étape qui a besoin de texte ET de médias fait UN SEUL appel au LLM qui prend les deux en entrée. Pas d'étape "décris l'image" suivie d'étape "classifie le texte".

### 3.4 Prompts hors du code

Tous les prompts métier vivent dans **Langfuse Prompt Management**. Le code fait `prompt_registry.get("pipeline/universe")`. En local, un fallback sur des fichiers YAML permet de dev sans connexion Langfuse. La version du prompt utilisée est loggée dans le span.

### 3.5 Structured outputs partout

Chaque appel LLM attend un schéma Pydantic. Utiliser les capacités de structured output natives (Gemini `response_schema`, OpenAI `response_format=json_schema`, Claude tool use). Fallback : JSON mode + validation Pydantic + retry.

### 3.6 Idempotence & traçabilité

Une pub traitée deux fois produit deux traces distinctes mais identifiables (même `platform_ad_id`). Langfuse session = un batch de traitement, trace = une pub, spans = les étapes.

### 3.7 Fail loud, fail clear

- Marque inconnue → `UnsupportedBrandError`.
- LLM qui renvoie du JSON invalide → retry N fois puis `LLMValidationError` avec le payload brut loggé.
- Média inaccessible → l'étape continue avec les médias restants, warning loggé, metadata du trace indique le média manquant.

---

## 4. Architecture cible

### 4.1 Arborescence du repo

```
project-root/
├── src/
│   └── pipeline_app/
│       ├── __init__.py
│       ├── config.py                 # Settings Pydantic (env vars)
│       ├── exceptions.py             # exceptions custom du domaine
│       │
│       ├── schemas/                  # Contrats de données
│       │   ├── __init__.py
│       │   ├── ad.py                 # Ad, AdText, AdMedia
│       │   ├── taxonomy.py           # BrandTaxonomy, TaxonomyNode
│       │   ├── products.py           # DetectedProduct, ProductClassification,
│       │   │                         # ProductName, FinalProductLabel
│       │   ├── pipeline.py           # PipelineInput, PipelineOutput, StepResult
│       │   └── scores.py             # ScoreReport
│       │
│       ├── data_access/              # Adapters I/O données
│       │   ├── __init__.py
│       │   ├── bigquery_client.py    # fetch_ad, fetch_ads_by_brand, fetch_batch
│       │   ├── media_fetcher.py      # download_media (retry + timeout)
│       │   └── results_repository.py # save_result, flag_for_review, get_result
│       │
│       ├── taxonomy/                 # Gestion taxonomies par marque
│       │   ├── __init__.py
│       │   ├── store.py              # TaxonomyStore: load/save JSON persistants
│       │   ├── loader.py             # TaxonomyLoader: point d'entrée unique
│       │   ├── chanel_deriver.py     # CSV -> taxonomy
│       │   └── generator.py          # LLM -> taxonomy (LV, Balenciaga)
│       │
│       ├── llm/                      # Abstraction LLM
│       │   ├── __init__.py
│       │   ├── base.py               # LLMProvider ABC + types communs
│       │   ├── gemini_provider.py
│       │   ├── openai_provider.py
│       │   ├── anthropic_provider.py (optionnel)
│       │   └── factory.py            # get_provider(name) -> LLMProvider
│       │
│       ├── observability/            # Langfuse
│       │   ├── __init__.py
│       │   ├── client.py             # singleton Langfuse client
│       │   ├── prompts.py            # PromptRegistry: fetch/cache prompts versionnés
│       │   ├── tracing.py            # decorators/helpers pour spans
│       │   └── scoring.py            # fonctions de scoring
│       │
│       ├── web_search/               # Fallback 5b
│       │   ├── __init__.py
│       │   ├── base.py               # WebSearchProvider ABC
│       │   ├── brand_domains.py      # mapping brand -> domaine officiel
│       │   └── tavily_provider.py    # (ou serper, ou brave)
│       │
│       ├── pipeline/                 # Logique métier pure
│       │   ├── __init__.py
│       │   ├── orchestrator.py       # ProductDetectionPipeline.run(ad) -> PipelineOutput
│       │   ├── step1_universe.py
│       │   ├── step2_products.py
│       │   ├── step3_classify.py
│       │   ├── step4_name.py
│       │   └── step5_fallback.py     # sub-steps 5a et 5b
│       │
│       ├── cli.py                    # CLI: process-ad, process-batch, refresh-taxonomy
│       └── main.py                   # entrypoint Docker
│
├── prompts/                          # Fallback local des prompts (YAML)
│   ├── universe.yaml
│   ├── products.yaml
│   ├── classification.yaml
│   ├── name_extraction.yaml
│   ├── fallback_enriched.yaml
│   ├── taxonomy_generation.yaml
│   └── judge.yaml                    # si LLM-as-judge
│
├── data/                             # Taxonomies & catalogue persistés
│   ├── taxonomies/
│   │   ├── chanel.json               # dérivée du CSV
│   │   ├── dior.json                 # fournie
│   │   ├── mfk.json                  # fournie
│   │   ├── louis_vuitton.json        # générée par LLM
│   │   └── balenciaga.json           # générée par LLM
│   └── catalogs/
│       └── chanel_products.csv       # source du catalogue Chanel
│
├── tests/
│   ├── unit/
│   │   ├── test_schemas.py
│   │   ├── test_taxonomy_loader.py
│   │   ├── test_chanel_deriver.py
│   │   └── test_scoring.py
│   ├── integration/
│   │   ├── test_pipeline_end_to_end.py  # avec LLM mocké
│   │   └── test_fallback.py
│   └── fixtures/
│       ├── sample_ads.json
│       └── mock_llm_responses.json
│
├── docs/
│   ├── DEPLOYMENT.md
│   ├── USAGE.md
│   └── screenshots/
│       ├── trace_full.png
│       ├── trace_with_fallback.png
│       ├── prompt_versioning.png
│       └── scores_dashboard.png
│
├── scripts/
│   ├── smoke.py                      # test d'infra (BQ + Langfuse + LLM)
│   └── seed_prompts.py               # push YAML -> Langfuse
│
├── Dockerfile
├── docker-compose.yml                # app + langfuse + postgres
├── .env.example
├── .gitignore
├── pyproject.toml                    # ou requirements.txt si pip classique
├── README.md
├── roadmap.md                        # CE DOCUMENT
└── Makefile                          # make up, make test, make refresh-taxo
```

### 4.2 Environnement Python

**Python 3.11+** recommandé (3.10 minimum).

Deux options de gestion de dépendances, choisies au début du projet :

**Option A — pip classique** (si tu veux rester sur ce que tu connais) :
- `requirements.txt` pour les deps runtime
- `requirements-dev.txt` pour pytest, ruff, etc.
- Venv créé à la main : `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

**Option B — uv** (recommandé si tu veux essayer, migration en 5 min) :
- `pyproject.toml` avec deps déclarées
- `uv add <package>` pour ajouter
- `uv sync` pour installer (crée le venv automatiquement)
- `uv run python script.py` pour exécuter sans activer le venv
- Lockfile `uv.lock` committé → reproductibilité garantie

Les deux marchent pour ce projet. L'important est de **figer les versions** (lockfile ou `pip freeze`).

---

## 5. Schémas de données

Tous dans `schemas/`. Tous en Pydantic v2.

### `schemas/ad.py`

- **`Brand`** : Enum — `CHANEL`, `DIOR`, `LOUIS_VUITTON`, `BALENCIAGA`, `MFK`. Toute valeur hors enum lève `UnsupportedBrandError`.
- **`AdText`** : `body_text: str | None`, `title: str | None`, `caption: str | None`, `url: str | None`.
- **`Ad`** : `platform_ad_id: str`, `brand: Brand`, `texts: list[AdText]`, `media_urls: list[str]`. Méthode helper `all_text() -> str` qui concatène title/body/caption/url.

### `schemas/taxonomy.py`

- **`BrandTaxonomy`** : structure arborescente `universe -> category -> subcategory -> product_type[]`. Stockée en JSON. Méthodes : `get_universes()`, `get_categories(universe)`, `is_valid_path(u, c, s, t) -> bool`, `serialize() / deserialize()`.

### `schemas/products.py`

- **`Color`** : Enum fermée (Black, White, Navy, Brown, Beige, Grey, Red, Blue, Green, Yellow, Purple, Pink, Orange, Gold, Silver, Multicolor).
- **`DetectedProduct`** (sortie étape 2) : `importance: int` (0-5), `color: Color`, `universe: str`, `raw_description: str` (pour réutilisation en étape 5a).
- **`ProductClassification`** (sortie étape 3) : `universe, category, subcategory, product_type, confidence: float`.
- **`ProductName`** (sortie étape 4 ou 5) : `name: str | None`, `source: Literal["explicit", "fallback_llm", "fallback_web"]`, `confidence: float`, `needs_review: bool`, `sources_consulted: list[str]`.
- **`FinalProductLabel`** : assemble `DetectedProduct` + `ProductClassification` + `ProductName`.

### `schemas/pipeline.py`

- **`PipelineOutput`** : `ad_id`, `brand`, `products: list[FinalProductLabel]`, `scores: ScoreReport`, `trace_id: str`.

### `schemas/scores.py`

- **`ScoreReport`** : dict typé contenant les scores calculés (coherence, confidence, judge...).

---

## 6. Couche LLM abstraite

### 6.1 L'interface `LLMProvider` (`llm/base.py`)

```
class LLMMessage       # text + list of media (URL ou bytes)
class LLMCallConfig    # model_name, temperature, max_tokens, timeout
class LLMResponse      # parsed (Pydantic instance), raw_json,
                       # usage (tokens in/out), latency_ms, model_used
class TraceContext     # pour attacher les calls à un span Langfuse

class LLMProvider(ABC):
    def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        config: LLMCallConfig,
        trace_context: TraceContext,
    ) -> LLMResponse: ...

    def generate_text(...): ...  # pour les cas sans schema (rare)

    @property
    def supports_video(self) -> bool: ...
    @property
    def name(self) -> str: ...
```

### 6.2 Comportement commun aux implémentations

Chaque implémentation (Gemini, OpenAI) :

1. Convertit `LLMMessage` vers le format natif du SDK.
2. Appelle le SDK avec structured output natif.
3. Parse la réponse en `response_model`.
4. Retry avec backoff sur erreurs transitoires (429, 500, timeout).
5. Retry sur parse error (2x max) avec message "le JSON précédent était invalide, voici l'erreur : X".
6. Log le call dans Langfuse via `trace_context`.

### 6.3 Choix du modèle par défaut

**Gemini 2.5 (via `google-genai`)** recommandé :
- Multimodal natif incluant vidéo (les autres modèles demandent d'extraire des frames).
- Supporte les URLs GCS directement.
- Structured output natif via `response_schema`.
- Prix/qualité excellent sur ce type de tâche.
- Déjà dans l'écosystème GCP du test.

**OpenAI GPT-4o** ou **Claude 3.5 Sonnet** comme alternatives, pour prouver l'abstraction.

---

## 7. Taxonomies

### 7.1 Point d'entrée unique

`taxonomy/loader.py` expose :

```
def load_taxonomy(brand: Brand, force_refresh: bool = False) -> BrandTaxonomy
```

Logique interne :
- **Dior / MFK** → lit le JSON fourni dans `data/taxonomies/`.
- **Chanel** → si `data/taxonomies/chanel.json` existe et `not force_refresh`, le charge ; sinon appelle `chanel_deriver.derive_from_csv()` et persiste.
- **LV / Balenciaga** → idem mais via `generator.generate_via_llm(brand)`.
- **Brand inconnue** → `UnsupportedBrandError`.

### 7.2 Stockage

`TaxonomyStore` gère la persistence : read/write JSON avec métadonnée `{generated_at, source: "provided"|"derived"|"llm"}`. Versioning optionnel avec timestamp dans le filename.

### 7.3 Rafraîchissement

CLI : `python -m pipeline_app.cli refresh-taxonomy --brand louis_vuitton` pour forcer une régénération. Indispensable pour démontrer que les taxos LLM peuvent évoluer sans toucher au code.

---

## 8. Le pipeline — étape par étape

### 8.1 Structure commune d'une étape

Chaque fichier `pipeline/stepN_xxx.py` suit la même structure :

```
def execute(
    <inputs spécifiques>,
    llm_provider: LLMProvider,
    prompt_registry: PromptRegistry,
    trace: TraceContext,
) -> <OutputSchema>:
    with trace.span(name="step_N_xxx") as span:
        prompt = prompt_registry.get("pipeline/stepN_xxx")
        messages = build_messages(prompt, <inputs>)
        response = llm_provider.generate_structured(
            messages=messages,
            response_model=<OutputSchema>,
            config=...,
            trace_context=span,
        )
        span.update(output=response.parsed)
        return response.parsed
```

### 8.2 Étape 1 — Universe identification

- **Input** : `Ad` complet (tous textes + tous médias).
- **Output** : `UniverseResult { detected_universes: list[Universe], reasoning: str, confidence: float }`.
- **Prompt** : insiste sur le fait qu'une pub peut couvrir plusieurs universes simultanément (ex. modèle portant un sac Chanel + pub parfum).

### 8.3 Étape 2 — Product identification

- **Input** : `Ad` + `UniverseResult`.
- **Output** : `list[DetectedProduct]`.
- **Prompt** : priorité texte d'abord (titres, body, caption, url), puis refinement visuel. Échelle d'importance 0-5. Chaque produit reçoit un `raw_description` qui sera réutilisé en étape 5a.

### 8.4 Étape 3 — Classification taxonomique

- **Input** : `DetectedProduct` + contexte pub + `BrandTaxonomy`.
- **Output** : `ProductClassification`.
- **Prompt** : injecte dynamiquement la taxonomie de la marque (format compact). Force le LLM à respecter la hiérarchie.
- **Validation post-LLM** : vérifier que `(universe, category, subcategory, product_type)` existe bien dans la taxonomie. Si non, retry avec "cette combinaison n'existe pas, voici les options valides pour ce category: [...]".

### 8.5 Étape 4 — Product name extraction

- **Input** : `DetectedProduct` + `ProductClassification` + contexte pub.
- **Output** : `ProductName | None`.
- **Prompt** : "renvoie le nom UNIQUEMENT si explicitement mentionné dans le texte ou clairement visible dans l'image. Renvoie `null` sinon." Important : le LLM doit être capable de dire "je ne sais pas".

### 8.6 Étape 5 — Fallback

Déclenchée si étape 4 retourne `None`. Voir §9.

---

## 9. Fallback de nommage

### 9.1 Sub-step 5a — LLM enrichi

Input constitué dynamiquement :
- `raw_description` du produit (accumulée depuis les étapes précédentes).
- Classification taxonomique complète.
- Couleur identifiée.
- **Si la marque a un catalogue** (Chanel) : on injecte la sous-partie pertinente du catalogue (filtré par universe + category + subcategory pour ne pas exploser le contexte).
- **Si pas de catalogue** : on demande au LLM de proposer sur la base de sa connaissance de la marque.

Output : `{suggested_name: str, confidence: float (0-1), reasoning: str}`.

### 9.2 Sub-step 5b — Web verification (conditionnelle)

- **Trigger** : `confidence < threshold_web_verify` (configurable, défaut 0.7).
- **Query** : `{brand} {suggested_name} {category}` avec filtre `site:{brand_official_domain}`.
- On fetch les 3-5 premiers résultats, on passe les snippets à un LLM qui confirme ou infirme.
- **Output** : `{confirmed: bool, final_name: str, source_url: str}`.

### 9.3 Décision finale

- `confidence >= threshold_accept` (défaut 0.85) → accepté.
- Sinon → `needs_review = True`, on garde la suggestion + score + sources.

### 9.4 Traçage

Tous les appels (LLM 5a, web search, LLM de vérif 5b) sont des spans enfants d'un span parent `"fallback"`. Le span racine de la trace pub voit : `step1 → step2 → [step3 → step4 → fallback(5a, 5b)] x N produits`.

---

## 10. Scoring & observabilité Langfuse

### 10.1 Scores implémentés

Minimum 2, idéalement 3 :

1. **Taxonomic coherence** (déterministe, pas de LLM) : le tuple `(universe, category, subcategory, product_type)` existe-t-il dans la taxonomie ? Binaire 0/1 ou fractionnel selon le niveau atteint. Log via `langfuse.score(name="taxonomy_coherence", value=...)`.
2. **Per-step confidence** : chaque étape LLM renvoie un `confidence` dans son schema output. On remonte ces scores.
3. **LLM-as-judge (bonus)** : un dernier appel LLM évalue la qualité du label final (cohérence globale, complétude, pertinence). Prompt dédié dans Langfuse.

### 10.2 Tracing

- Un **client Langfuse singleton** créé au démarrage depuis les env vars.
- Chaque run pipeline crée une **trace racine** avec `ad.platform_ad_id`.
- Chaque étape crée un **span enfant**.
- Chaque appel LLM crée une **generation enfant** du span d'étape, avec `input`, `output`, `model`, `usage`, `prompt_version`.
- **Sessions** : la CLI batch crée un `session_id` propagé à toutes les traces du batch.
- **Tags** : `[brand, platform]` minimum.

### 10.3 Prompts

Tous dans Langfuse Prompt Management. `PromptRegistry.get(name, label="production")` renvoie un objet avec `.compile(**vars)` et `.version`. Cache local 5 min. Fallback sur YAML local si Langfuse unreachable.

Script d'init `scripts/seed_prompts.py` pour push les YAML vers Langfuse au premier déploiement.

---

## 11. Roadmap d'implémentation par phases

### Phase 0 — Bootstrap & infra (priorité absolue)

**Objectif** : projet qui démarre, Langfuse reçoit une trace dummy, BigQuery renvoie une pub.

- Init repo, `pyproject.toml` (ou `requirements.txt`). Deps : `pydantic`, `pydantic-settings`, `google-cloud-bigquery`, `langfuse`, `httpx`, `google-genai`, `openai`, `python-dotenv`, `pytest`, `typer`.
- Structure de dossiers complète (tous les `__init__.py` et stubs).
- `config.py` : `Settings(BaseSettings)` lit `.env`. Variables : `GCP_PROJECT_ID`, `LLM_PROVIDER`, `LLM_MODEL`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `WEB_SEARCH_API_KEY`, `FALLBACK_CONFIDENCE_THRESHOLD`, `WEB_VERIFY_THRESHOLD`.
- `.env.example` avec toutes les variables.
- `docker-compose.yml` : services Langfuse v3 self-host (`langfuse-web`, `langfuse-worker`, `postgres`, `clickhouse`, `minio`, `redis`) + service `app`.
- `Dockerfile` de l'app : `python:3.11-slim`, install deps, copy src, entrypoint.
- `scripts/smoke.py` : envoie une trace Langfuse vide + requête BQ `SELECT COUNT(*)` + appel Gemini "dis bonjour".

**Checkpoint** : `make smoke` passe. Trace visible dans Langfuse UI.

### Phase 1 — Schemas & data access

**Objectif** : lire une pub BQ, télécharger ses médias, tout est typé.

- Tous les schemas de `schemas/` (§5). Tests unitaires.
- `exceptions.py` : `UnsupportedBrandError`, `LLMValidationError`, `MediaFetchError`, `TaxonomyNotFoundError`.
- `data_access/bigquery_client.py` : `fetch_ad`, `fetch_ads_by_brand`, `fetch_batch`.
- `data_access/media_fetcher.py` : `download(url)` avec timeout, retry, size limit.
- `data_access/results_repository.py` : abstrait, impl JSON files ou SQLite.

**Checkpoint** : un script récupère une pub via BQ, télécharge ses médias, affiche les types.

### Phase 2 — Abstraction LLM

**Objectif** : appel multimodal structuré via interface agnostique.

- `llm/base.py` : `LLMProvider`, `LLMMessage`, `LLMCallConfig`, `LLMResponse`, `TraceContext`.
- `llm/gemini_provider.py` : SDK `google-genai`, conversion messages, `response_schema` natif, retries, logging Langfuse.
- `llm/openai_provider.py` : SDK `openai`, `response_format=json_schema`, médias base64.
- `llm/factory.py` : `get_provider(name)`.
- Tests unitaires (SDK mocké) + un test d'intégration live désactivable en CI.

**Checkpoint** : `provider.generate_structured(...)` marche avec Gemini et renvoie une instance Pydantic validée.

### Phase 3 — Taxonomies

**Objectif** : `BrandTaxonomy` disponibles pour toutes les marques, via la bonne source.

- `taxonomy/store.py` : load/save JSON + métadonnée.
- `taxonomy/chanel_deriver.py` : CSV → arbre taxonomique.
- `taxonomy/generator.py` : LLM → taxo (LV, Balenciaga).
- `taxonomy/loader.py` : point d'entrée, dispatch par marque.
- CLI : `refresh-taxonomy --brand <brand>`.

**Checkpoint** : `load_taxonomy(Brand.CHANEL)` renvoie une taxo cohérente. Idem pour LV après refresh LLM.

### Phase 4 — Observability (Langfuse wiring)

**Objectif** : infra de tracing prête AVANT l'implémentation des étapes.

- `observability/client.py` : singleton `get_langfuse()`.
- `observability/prompts.py` : `PromptRegistry.get()` avec cache et fallback YAML.
- `observability/tracing.py` : context managers `pipeline_trace(ad)` et `step_span(trace, name)`.
- `observability/scoring.py` : `score_taxonomy_coherence`, `score_confidence`, `score_llm_judge`.
- Écrire les YAML initiaux dans `prompts/`.
- `scripts/seed_prompts.py`.

**Checkpoint** : test qui ouvre trace → span → log score → on voit tout dans Langfuse.

### Phase 5 — Pipeline steps (le cœur)

**Objectif** : chaque étape fonctionne isolément et produit le bon output.

- `step1_universe.py` / `step2_products.py` / `step3_classify.py` / `step4_name.py` (§8).
- Chacune : un test unitaire (LLM mocké) + un test d'intégration sur une pub réelle.

**Checkpoint** : chaque étape passe ses tests.

### Phase 6 — Orchestrator & CLI

**Objectif** : traiter une pub ou un batch via une commande.

- `pipeline/orchestrator.py` : `ProductDetectionPipeline.run(ad)`. Dependency injection (pas de singletons internes). Gestion d'erreurs par étape.
- `step5_fallback.py` intégré.
- `cli.py` avec `typer` : `process-ad`, `process-batch`, `refresh-taxonomy`, `seed-prompts`.
- `main.py` : entrypoint Docker.

**Checkpoint** : `docker compose run app process-ad --ad-id X` traite une pub, persiste, trace visible dans Langfuse.

### Phase 7 — Scoring complet & bonus

- Les 2-3 scores de §10.1.
- **LLM-as-judge** (bonus).
- **Tags Langfuse** : `brand`, `platform`.
- Validation que les sessions groupent bien les batches.

### Phase 8 — Documentation & packaging final

- `README.md` : approche, choix d'archi, justification modèles, fallback, taxonomies, Langfuse, limites, **temps passé honnête**.
- `docs/DEPLOYMENT.md` : prérequis, `gcloud auth`, `.env`, `docker compose up`, init Langfuse, seed prompts, refresh taxos, premier run.
- `docs/USAGE.md` : pour chaque fonction publique — signature, description, exemple call/response.
- **Screenshots Langfuse** : trace complète, trace avec fallback, prompt versioning, dashboard cost/latency, score distribution.

### Phase 9 — Tests & robustesse (en continu)

- Tests unitaires : schemas, deriver/loader taxo, scoring.
- Tests d'intégration : pipeline avec LLM mocké (FakeLLMProvider avec réponses canned).
- Un test end-to-end sur une vraie pub, marqué `@pytest.mark.live`.

---

## 12. Règles de travail (assistant de code)

À appliquer systématiquement par tout assistant de code (Claude Code, Cursor, etc.) travaillant sur ce projet.

### 12.1 Respect de l'architecture

1. **Suivre les phases 0 → 9 dans l'ordre.** Ne pas passer à la phase N+1 sans avoir validé le checkpoint de la phase N.
2. **Respecter strictement la séparation de couches** (schemas → domain → adapters → orchestration) décrite en §3.1 et §4.
3. **Tout ce qui entre ou sort d'une couche est typé Pydantic.** Pas de `dict` ou `Any` en signature publique.
4. **Aucun import direct de SDK LLM hors du dossier `llm/`.** Aucun import `google.cloud.bigquery` hors de `data_access/`. Aucun import `langfuse` hors de `observability/`.
5. **Les prompts métier vivent dans des fichiers YAML (`prompts/`) et sont push dans Langfuse via un script.** Jamais hardcodés dans le code Python.

### 12.2 Docstrings obligatoires

**Chaque fonction (et méthode publique) DOIT avoir une docstring standard comprenant** :

- **2 à 3 phrases maximum** décrivant le use case / ce que la fonction fait.
- **Les entrées** : chaque paramètre listé avec son type et son rôle.
- **La sortie** : ce qui est retourné, avec son type.
- **Les exceptions levées** si pertinent.

Format Google-style ou NumPy-style, au choix, mais **cohérent dans tout le projet**. Exemple Google-style :

```python
def fetch_ad(ad_id: str) -> Ad:
    """Récupère une pub depuis BigQuery par son identifiant.

    Utilisée par la CLI `process-ad` et par le pipeline lorsqu'on traite
    une pub unique plutôt qu'un batch.

    Args:
        ad_id: L'identifiant unique de la pub (colonne `platform_ad_id`).

    Returns:
        L'objet `Ad` validé, avec textes et URLs de médias.

    Raises:
        AdNotFoundError: Si aucune ligne ne correspond à `ad_id`.
        UnsupportedBrandError: Si la marque de la pub n'est pas supportée.
    """
```

Cette règle est **non-négociable**. Elle garantit que le code reste lisible et documenté automatiquement via Sphinx/mkdocs plus tard.

### 12.3 Flux de travail

6. **Avant d'écrire du code pour une phase, lister le plan d'attaque** (fichiers à créer, deps à installer, tests à écrire). Attendre validation explicite de l'humain avant de coder.
7. **Chaque phase produit un commit** avec message clair (`feat(phase-3): chanel taxonomy deriver`).
8. **Tourner les tests avant chaque commit.** `pytest` doit passer.
9. **Si un choix de design n'est pas spécifié, expliquer en 2 lignes pourquoi on prend l'option X et continuer.** Ne pas bloquer.

### 12.4 Qualité de code

10. Python 3.11+, typage strict (`from __future__ import annotations` autorisé).
11. **Ruff** pour le lint (`ruff check` + `ruff format`).
12. **Pytest** pour les tests. Fixtures dans `tests/fixtures/`.
13. Pas de `print()` en prod — utiliser le logging standard.
14. **Error handling explicite** : pas de `except Exception: pass`. Toujours logger.

### 12.5 Sécurité & config

15. Jamais de clé API committée. Tout vient de `.env` (ignoré par git).
16. `.env.example` à jour avec TOUTES les variables, sans valeurs sensibles.
17. Les logs ne doivent pas contenir de clés API ni de PII.

---

## 13. Questions ouvertes à clarifier

À envoyer par mail au jury **avant de démarrer l'implémentation** (l'énoncé valorise explicitement la capacité à poser des questions pertinentes).

1. **Taxonomie de référence** : le fichier `product_categorisation.json` fourni dans le projet a des universes (`Women`, `Men`, `Unisex`, `Beauty`, `Home`, `Watches & Jewelry`, `Technology`, `Kids & Baby`, etc.) qui ne correspondent pas à ceux de l'énoncé (`Fashion`, `Beauty`, `Home`, `Insurance`). Quelle est la source de vérité ? Le JSON existant est-il un point de départ à adapter, ou faut-il l'ignorer au profit de la taxonomie de l'énoncé ?

2. **Universe "Insurance"** : pour des marques de luxe (Chanel, Dior, LV, Balenciaga, MFK), y a-t-il réellement des cas "Insurance" dans le dataset ? Sinon, je peux supporter sa détection mais ne pas la prioriser.

3. **Taxonomies fournies (`dior_taxonomy.json`, `mkf_taxonomy.json`)** : ces fichiers ne sont pas présents dans les fichiers reçus. Où les trouver ?

4. **Catalogue Chanel (`chanel_products_taxonomy.csv`)** : même question, pas dans les fichiers fournis — à partager.

5. **Gestion des vidéos** : les `media_urls` contiennent-elles des vidéos ? Si oui, préférez-vous un modèle qui les gère nativement (Gemini), ou l'extraction de frames clés ?

6. **Stockage des résultats** : schéma cible libre ? Je propose JSON files ou SQLite par défaut (cohérent avec l'infra Docker locale). Acceptable ?

7. **Field `platform`** dans les données : existe-t-il ? Mentionné dans la roadmap pour les tags Langfuse mais pas dans le schéma BQ décrit dans l'énoncé.

---

**Fin du document. Ce fichier doit être maintenu à jour au fur et à mesure des décisions prises pendant l'implémentation.**