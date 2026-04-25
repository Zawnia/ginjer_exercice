from __future__ import annotations

from pathlib import Path
import shutil

import yaml

from ginjer_exercice.observability.prompts import PromptRegistry


def test_all_prompts_use_max_tokens_4000() -> None:
    prompts_dir = Path("prompts")
    for prompt_file in prompts_dir.glob("*.yaml"):
        with prompt_file.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
        assert content["config"]["max_tokens"] == 4000, prompt_file.name


def test_langfuse_prompt_uses_local_yaml_max_tokens(monkeypatch) -> None:
    prompts_dir = Path("output/test_prompts/prompts")
    if prompts_dir.parent.exists():
        shutil.rmtree(prompts_dir.parent)
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "classification.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "pipeline/classification",
                "config": {"model": "gemini-2.0-flash", "temperature": 0.1, "max_tokens": 4000},
                "prompt": "Classify {{ product_description }}",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyPrompt:
        version = 1
        prompt = "Remote prompt"
        config = {"model": "gemini-2.5-flash", "temperature": 0, "max_tokens": 1024}

    class DummyLangfuse:
        def get_prompt(self, name: str, label: str, type: str):
            return DummyPrompt()

    monkeypatch.setattr("ginjer_exercice.observability.prompts.get_langfuse_client", lambda: DummyLangfuse())

    prompt = PromptRegistry(prompts_dir=prompts_dir).get("pipeline/classification")

    assert prompt.source == "langfuse"
    assert prompt.prompt == "Remote prompt"
    assert prompt.config["max_tokens"] == 4000
