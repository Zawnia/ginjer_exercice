# ginjer_exercice

Pipeline de détection et labellisation produit pour publicités luxe, avec lecture BigQuery, exécution LLM par steps, et persistance SQLite.

## Installation rapide

```bash
uv sync
cp .env.example .env
```

Variables minimales utiles :
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GCP_PROJECT_ID`
- `LLM_PROVIDER`
- `GEMINI_API_KEY` ou `OPENAI_API_KEY`
- `SQLITE_DB_PATH`

## Commandes

Rafraîchir la taxonomie canonique :

```bash
uv run python -m ginjer_exercice.cli refresh-taxonomy
```

Traiter une pub par `ad_id` :

```bash
uv run python -m ginjer_exercice.cli process-ad <AD_ID>
```

Avec une base SQLite explicite :

```bash
uv run python -m ginjer_exercice.cli process-ad <AD_ID> --db-path data/results/pipeline_results.db
```

## Architecture

Le flux produit P0 est :

```text
BigQuery -> Ad -> step1 universe -> step2 products -> step3 classify -> step4 name -> PipelineOutput -> SQLite
```

Principes retenus :
- les steps restent découplés et testables
- l’orchestrateur assemble le flux produit minimal
- `ResultsRepository` est la frontière de persistance
- Langfuse reste best-effort et ne doit pas bloquer l’exécution

## Limitations & Extensions

État réel livré en P0 :
- `step2` est multimodal images seulement
- les vidéos sont ignorées dans `step2`
- `step5` fallback n’est pas implémenté
- `process-batch` n’est pas implémenté
- `llm_judge` n’est pas implémenté
- OpenAI reste sans vidéo
- l’intégration Langfuse est utile mais pas totalement alignée avec l’architecture racontée dans `.memory`

Compléments :
- docs/USAGE.md
- docs/DEPLOYMENT.md
