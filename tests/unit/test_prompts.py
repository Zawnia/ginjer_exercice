from __future__ import annotations

from pathlib import Path

import yaml


def test_all_prompts_use_max_tokens_4000() -> None:
    prompts_dir = Path("prompts")
    for prompt_file in prompts_dir.glob("*.yaml"):
        with prompt_file.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
        assert content["config"]["max_tokens"] == 4000, prompt_file.name
