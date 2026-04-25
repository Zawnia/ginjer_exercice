from typing import Any, Literal
from pathlib import Path
import yaml
import logging
import time

from pydantic import BaseModel, Field

from .client import get_langfuse_client
from ..config import get_settings

logger = logging.getLogger(__name__)

class ManagedPrompt(BaseModel):
    """Prompt résolu, prêt à l'emploi par le pipeline.
    
    Abstraction interne qui découple le code métier de l'objet SDK Langfuse.
    Le champ `config` permet de versionner les paramètres modèle
    (temperature, max_tokens, etc.) avec le prompt dans Langfuse.
    """
    name: str
    version: str | None = None
    label: str | None = None
    prompt: str
    config: dict[str, Any] = Field(default_factory=dict)
    source: Literal["langfuse", "yaml_fallback"] = "langfuse"

    def compile(self, **variables: str) -> str:
        """Remplace les variables {{var}} dans le prompt."""
        text = self.prompt
        for key, value in variables.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
        return text

class PromptRegistry:
    """Registre centralisé des prompts avec cache et fallback.
    
    Stratégie de résolution :
        1. Cache mémoire (TTL configurable)
        2. Langfuse (get_prompt avec label)
        3. Fallback YAML local (prompts/{name}.yaml)
    """
    
    def __init__(self, cache_ttl: int | None = None, prompts_dir: Path | None = None):
        settings = get_settings()
        self.cache_ttl = cache_ttl if cache_ttl is not None else settings.prompt_cache_ttl_seconds
        
        if prompts_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.prompts_dir = base_dir / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
            
        self._cache: dict[str, dict[str, Any]] = {}  # key: f"{name}:{label}", value: {"prompt": ManagedPrompt, "timestamp": float}

    def _get_cache_key(self, name: str, label: str) -> str:
        return f"{name}:{label}"

    def _get_from_cache(self, key: str) -> ManagedPrompt | None:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] <= self.cache_ttl:
                return entry["prompt"]
        return None

    def _put_in_cache(self, key: str, prompt: ManagedPrompt):
        self._cache[key] = {
            "prompt": prompt,
            "timestamp": time.time()
        }

    def _load_yaml_data(self, name: str) -> dict[str, Any] | None:
        filename = name.split("/")[-1] + ".yaml"
        filepath = self.prompts_dir / filename
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, name: str, label: str = "production") -> ManagedPrompt:
        """Récupère un prompt par nom et label."""
        cache_key = self._get_cache_key(name, label)
        cached_prompt = self._get_from_cache(cache_key)
        if cached_prompt:
            return cached_prompt

        # 1. Tenter Langfuse
        langfuse = get_langfuse_client()
        if langfuse is not None:
            try:
                lf_prompt = langfuse.get_prompt(name, label=label, type="text")
                config = lf_prompt.config if hasattr(lf_prompt, 'config') else {}
                if config is None:
                    config = {}
                yaml_data = self._load_yaml_data(name)
                if yaml_data is not None:
                    yaml_config = yaml_data.get("config", {})
                    if "max_tokens" in yaml_config:
                        config = dict(config)
                        config["max_tokens"] = yaml_config["max_tokens"]
                
                managed = ManagedPrompt(
                    name=name,
                    version=str(lf_prompt.version) if hasattr(lf_prompt, 'version') else None,
                    label=label,
                    prompt=lf_prompt.prompt,
                    config=config,
                    source="langfuse"
                )
                self._put_in_cache(cache_key, managed)
                return managed
            except Exception as e:
                logger.warning(f"Impossible de récupérer le prompt '{name}' depuis Langfuse: {e}. Fallback YAML.")

        # 2. Fallback YAML
        return self._get_from_yaml(name, label)

    def _get_from_yaml(self, name: str, label: str) -> ManagedPrompt:
        """Charge le prompt depuis les fichiers YAML locaux."""
        data = self._load_yaml_data(name)
        filename = name.split("/")[-1] + ".yaml"
        filepath = self.prompts_dir / filename

        if data is None:
            raise FileNotFoundError(f"Le prompt '{name}' n'est pas dans Langfuse et le fichier fallback {filepath} n'existe pas.")

        managed = ManagedPrompt(
            name=name,
            version=None,
            label=label,
            prompt=data.get("prompt", ""),
            config=data.get("config", {}),
            source="yaml_fallback"
        )
        
        # On met en cache aussi le fallback pour éviter de lire le disque en boucle
        self._put_in_cache(self._get_cache_key(name, label), managed)
        return managed
