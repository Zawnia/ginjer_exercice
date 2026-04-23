# Deployment

## Configuration

Le pipeline dépend de :
- accès BigQuery en lecture
- un provider LLM configuré
- une base SQLite locale ou montée en volume
- Langfuse optionnel

Variables d’environnement principales :

```env
GOOGLE_APPLICATION_CREDENTIALS=./.secret/gcp-sa.json
GCP_PROJECT_ID=ginjer-440122
LLM_PROVIDER=gemini
GEMINI_API_KEY=
OPENAI_API_KEY=
SQLITE_DB_PATH=data/results/pipeline_results.db
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_ENABLED=true
```

## SQLite

La persistance P0 repose sur `SQLiteResultsRepository`.

Par défaut :

```text
data/results/pipeline_results.db
```

La commande `process-ad` crée le répertoire parent si nécessaire.

## Langfuse self-host

Le repo prévoit une intégration Langfuse best-effort :
- prompts versionnés via `PromptRegistry`
- traces pipeline et steps
- publication des scores minimaux quand un `trace_id` est disponible

État réel à garder en tête :
- l’observabilité ne doit pas faire échouer le pipeline
- l’intégration actuelle n’est pas encore une implémentation “finalisée” de toute l’architecture décrite dans `.memory`
- cette doc ne suppose pas de conformité complète v4 au-delà du flux utile à `process-ad`

## Limites P0

- pas de fallback llm / web search
- pas de `process-batch`
- pas de `llm_judge`
- multimodal partiel, images seulement sur `step2`
- OpenAI sans vidéo
