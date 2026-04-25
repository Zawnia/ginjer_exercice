# ginjer_exercice

Pipeline de détection et de labellisation produit dans des publicités de marques de luxe : Chanel, Dior, Louis Vuitton, Balenciaga, Maison Francis Kurkdjian.

Lecture depuis BigQuery, orchestration LLM en cinq étapes, observabilité Langfuse, persistance SQLite.

## Quickstart

```bash
uv sync
cp .env.example .env
# éditer .env avec vos credentials
```

Variables minimales :

- `GOOGLE_APPLICATION_CREDENTIALS` — credentials GCP BigQuery + Vertex AI
- `GCP_PROJECT_ID`
- `LLM_PROVIDER` — `gemini` par défaut, ou `openai`
- `GEMINI_API_KEY` ou `OPENAI_API_KEY`
- `SQLITE_DB_PATH` — destination SQLite pour les résultats
- Variables Langfuse optionnelles, voir `docs/DEPLOYMENT.md`

Traiter une publicité :

```bash
uv run python -m ginjer_exercice.cli process-ad 1000142752321014
```

Rafraîchir la taxonomie canonique depuis le JSON source :

```bash
uv run python -m ginjer_exercice.cli refresh-taxonomy
```

## Exemple de sortie

Run réel sur une publicité Chanel (`1000142752321014`) :

```text
[STEP 1] Detection des univers...
-> Fragrance (conf: 1.00)
-> Women (conf: 0.90)
-> Luxury (conf: 0.80)
-> Beauty (conf: 0.70)

[STEP 2] Detection des produits...
-> 3 produit(s) detecte(s)

[PRODUIT 1/3] Flacon iconique du parfum CHANEL N°5...
[OK] Parfum > Parfum > Eaux de Parfum & Extraits (conf: 0.95)
[OK] 'CHANEL N°5' (source: explicit, conf: 1.00)

[PRODUIT 2/3] Sac à main CHANEL 2.55 en cuir matelassé noir...
[OK] Mode > Maroquinerie > Sacs (conf: 1.00)
[OK] 'CHANEL 2.55' (source: explicit, conf: 0.95)

[PRODUIT 3/3] Rouge à lèvres CHANEL Rouge Allure...
[OK] Maquillage > Lèvres > Rouge à lèvres (conf: 0.98)
[OK] 'Rouge Allure' (source: explicit, conf: 1.00)

Status: clean
Duration: 49.2s | LLM calls: 8 | Products: 3 | Taxonomy valid: 3/3
```

Chaque run est tracé dans Langfuse avec spans nestés, generations, usage tokens et coûts. Voir section Observabilité.

## Architecture

Pipeline en couches strictement séparées :

```text
BigQuery -> Ad -> [step1 universe] -> [step2 products]
                                      |
                                      v
                    pour chaque produit :
                    [step3 classify] -> [step4 name]
                                             |
                                             v
                                      si pas de nom :
                                      [step5 fallback]
                                             |
                                             v
                                PipelineOutput -> SQLite
```

Organisation du code :

| Couche | Responsabilité | Dépendance externe |
|--------|----------------|--------------------|
| `schemas/` | Pydantic | — |
| `pipeline/` | Logique métier des steps + orchestrator | aucune |
| `llm/` | Abstraction `LLMProvider`, impl Gemini + OpenAI | SDK google-genai, openai |
| `data_access/` | BigQuery, MediaFetcher, ResultsRepository | google-cloud-bigquery, httpx, sqlite3 |
| `observability/` | TraceContext wrapper Langfuse, PromptRegistry | Langfuse SDK |
| `web_search/` | WebSearchProvider (pas implémenté) | à venir |


## Choix de design

**LLM model-agnostic via factory pattern.** L'interface `LLMProvider` expose `generate_structured(messages, response_model, config, trace_context)`. Deux implémentations : Gemini par Vertex AI et OpenAI. Changer de modèle se fait par variable d'environnement, sans toucher au pipeline métier.

**Structured outputs Pydantic à chaque appel LLM.** Le schema de sortie est passé au provider, qui le transmet au SDK (`response_schema` côté Gemini, `response_format` côté OpenAI). Cela garantit le contrat sans parsing défensif dans les steps.

**Multimodal ciblé sur step2.** Gemini a été choisi pour son support natif des inputs images et vidéo. Dans ce livrable, seul `step2`, détection produit, injecte les images, car c'est le step qui en bénéficie le plus pour l'identification visuelle du produit. Les autres steps fonctionnent en text-only. Le contrat `LLMMessage.parts` (`TextPart | MediaPart`) est déjà en place pour étendre sans refactorer.

**Taxonomies hiérarchiques par marque.** Dior et MFK sont chargées depuis les fichiers fournis. Chanel est dérivée du JSON  `product_categorisation`. LV et Balenciaga seront générées via LLM, avec cache disque. La classification en step3 est contrainte par la taxonomie de la marque : impossible de classifier dans une catégorie hors taxonomie.

**Prompts versionnés dans Langfuse.** Chaque prompt métier vit dans Langfuse Prompt Management avec fallback YAML local. Aucun prompt hardcodé. Cela permet d'itérer sur les prompts sans redéployer.

**Signal qualité explicite.** Chaque `PipelineOutput` expose `quality_status` (`clean` ou `degraded`) et une liste de `warnings`. Un run peut être fonctionnellement réussi mais marqué `degraded` : fallback texte seul sur step2 faute d'image exploitable, réparation JSON d'une réponse LLM malformée, etc. 

## Observabilité

Langfuse trace chaque run avec la hiérarchie suivante :

```text
session (batch CLI)
└── trace (pipeline_{ad_id})
    ├── span step_1_universe
    │   └── generation llm_gemini-2.5-flash (tokens + cost)
    ├── span step_2_products
    │   └── generation llm_gemini-2.5-flash
    └── (×N produits)
        ├── span step_3_classify
        │   └── generation
        └── span step_4_name
            └── generation
```

versionning des prompts + input/output + couts tokens tracés par langfuse

Screenshots dans `docs/images/`.

## Limitations assumées

Scope volontairement borné pour ce livrable :

- **step5 fallback non implémenté.** Quand step4 ne trouve pas de nom explicite, le produit est persisté avec `name_info: None` et marqué `needs_review`. L'archi prévoit un fallback 5a LLM enrichi avec catalogue marque + 5b vérification web via `WebSearchProvider`, non codé par arbitrage temps.
- **Multimodal limité à step2.** Les autres steps ne reçoivent pas les images. Étendre demande uniquement de passer `media_parts` au message dans le step concerné, sans refactor.
- **Vidéos ignorées dans step2.** Les ads avec média vidéo passent en text-only avec un warning. Gemini supporte la vidéo nativement, mais le routage n'est pas activé pour ce livrable.
- **Gemini peut réparer du JSON malformé.** En cas de réponse LLM non parseable, une tentative de réparation est faite. Un warning est alors ajouté au `PipelineOutput`.
- **process-batch non implémenté.** Seul `process-ad`, une pub à la fois, est exposé en CLI.
- **LLM-as-judge non implémenté.** `scoring.py` calcule `taxonomy_coherence` et `confidence` agrégée. Le jugement LLM est stubbed.
- **OpenAI provider sans support vidéo** par limitation SDK.
- **URLs média `gs://` non téléchargées** par le fetcher dans ce flux. Les ads concernées passent en text-only avec un warning.
- **Catalogue** : l'énoncé mentionne un CSV `chanel_products_taxonomy.csv` (~320 entrées avec noms produits), mais celui-ci n'a pas été fourni dans les livrables du test (confirmé avec l'équipe Ginjer). Le `CatalogProvider` s'appuie 
donc sur la taxonomie canonique persistée dans `data/taxonomies/canonical.json`. Le cross-referencing de 5a se fait sur les catégories/sous-catégories disponibles ; l'interface `CatalogProvider` permet de swapper vers un `ChanelCSVCatalogProvider` sans changer le pipeline si le catalogue venait à être disponible.

## Extensions possibles

- **step5 fallback** : interface stable, orchestrator câblé pour le router quand `name_info is None`.
- **Multimodal étendu** : injection des images dans les messages des autres steps.
- **process-batch** : itération sur un set d'`ad_ids`, sessionnage Langfuse déjà prévu.
- **LLM-as-judge** : emplacement réservé dans `ScoringResult`, prompt de judge à écrire.
- **Nouveau provider LLM** : implémenter `LLMProvider`, déclarer dans la factory.
- **Nouvelles marques** : ajouter un fichier taxonomie ou générer via `refresh-taxonomy`.

## Documentation complémentaire

- `docs/USAGE.md` — usage détaillé, flux de données, signal qualité
- `docs/DEPLOYMENT.md` — déploiement, Langfuse self-host, configuration

