from __future__ import annotations

from pathlib import Path
import shutil

from typer.testing import CliRunner

from ginjer_exercice.cli import app
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.pipeline import PipelineOutput
from ginjer_exercice.schemas.scores import ScoreReport

runner = CliRunner()


def test_process_ad_fetches_and_runs_orchestrator(monkeypatch) -> None:
    captured = {}
    workspace_tmp = Path("output/test_cli")
    if workspace_tmp.exists():
        shutil.rmtree(workspace_tmp)
    workspace_tmp.mkdir(parents=True, exist_ok=True)

    class DummyBqClient:
        def fetch_ad(self, ad_id: str) -> Ad:
            captured["ad_id"] = ad_id
            return Ad(
                platform_ad_id=ad_id,
                brand=Brand.CHANEL,
                texts=[AdText(title="Campaign")],
                media_urls=[],
            )

    class DummyRepository:
        pass

    class DummyBigQueryFactory:
        def __init__(self, project: str):
            self.project = project

    monkeypatch.setattr("ginjer_exercice.cli.get_settings", lambda: type("S", (), {
        "sqlite_db_path": str(workspace_tmp / "results.db"),
        "google_application_credentials": None,
        "gcp_project_id": "demo-project",
        "llm_provider": "gemini",
        "openai_api_key": None,
    })())
    monkeypatch.setattr("ginjer_exercice.cli.bigquery.Client", DummyBigQueryFactory)
    monkeypatch.setattr("ginjer_exercice.cli.BigQueryClient", lambda bq_client: DummyBqClient())
    monkeypatch.setattr("ginjer_exercice.cli.get_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr("ginjer_exercice.cli.PromptRegistry", lambda: object())
    monkeypatch.setattr("ginjer_exercice.cli.SQLiteResultsRepository", lambda conn: DummyRepository())
    monkeypatch.setattr(
        "ginjer_exercice.cli.run_ad",
        lambda ad, **kwargs: PipelineOutput(
            ad_id=ad.platform_ad_id,
            brand=ad.brand,
            products=[],
            scores=ScoreReport(taxonomy_coherence=0.0, confidence=0.0, llm_judge=None),
            trace_id="trace-123",
        ),
    )

    result = runner.invoke(app, ["process-ad", "ad-42", "--db-path", str(workspace_tmp / "custom.db")])

    assert result.exit_code == 0
    assert captured["ad_id"] == "ad-42"
    assert "ad_id: ad-42" in result.stdout
    assert "trace_id: trace-123" in result.stdout
    assert f"sqlite_path: {workspace_tmp / 'custom.db'}" in result.stdout
