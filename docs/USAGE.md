# Usage

## Flux P0

La commande `process-ad` exécute le flux suivant :

```text
BigQuery.fetch_ad(ad_id)
  -> orchestrator.run_ad(ad)
  -> step1_universe
  -> step2_products
  -> step3_classify (par produit)
  -> step4_name (par produit)
  -> ScoreReport minimal
  -> SQLiteResultsRepository.save
```

Le résultat persisté est un `PipelineOutput`.

## Rôle des steps

- `step1_universe` détecte les univers probables à partir du texte et des médias.
- `step2_products` détecte les produits visibles. En P0, c’est le seul step qui télécharge réellement des images.
- `step3_classify` classe chaque produit dans la taxonomie de la marque.
- `step4_name` extrait un nom explicite s’il est présent dans le contenu. Si aucun nom n’est trouvé, `name_info=None`.

## Multimodal P0

Le multimodal livré est volontairement partiel :
- `step2` télécharge les médias via `MediaFetcher`
- seules les images sont injectées au LLM
- les vidéos sont ignorées avec logs explicites
- si aucune image ne peut être exploitée, `step2` continue en texte seul
- les autres steps conservent leur comportement actuel

## Scores minimaux

Le `ScoreReport` produit par l’orchestrateur contient :
- `taxonomy_coherence` : moyenne de validité des chemins taxonomiques des produits finalisés
- `confidence` : moyenne par produit des confiances disponibles (`classification.confidence`, puis `name_info.confidence` si présent)
- `llm_judge` : `None`

## needs_review

`PipelineOutput.needs_review` vaut `True` si au moins un produit :
- n’a pas de `name_info`
- ou a `name_info.needs_review=True`

En P0, l’absence de nom explicite n’est pas une erreur d’exécution. C’est un résultat incomplet assumé.

## Exemple

```bash
uv run python -m ginjer_exercice.cli process-ad 1234567890
```

Résumé affiché :
- `ad_id`
- `brand`
- nombre de produits
- nombre de noms trouvés
- `needs_review`
- `trace_id`
- chemin SQLite
