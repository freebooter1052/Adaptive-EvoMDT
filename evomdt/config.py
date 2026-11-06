from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import RoleName, SPECIALIST_ROLES


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines into the environment. Values already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.split(" #")[0].strip().strip("\"'"))


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key_env: str | None = None
    api_key: str | None = None
    base_url: str
    model_name: str
    temperature: float | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def resolve_api_key(self) -> "LLMConfig":
        if self.api_key is None and self.api_key_env:
            self.api_key = os.getenv(self.api_key_env)
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_llm: str
    artifacts_dir: str = "artifacts"
    evolution_state_path: str = "artifacts/evolution_state.json"


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: str
    weight: float | None = Field(default=None, gt=0.0)


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parallel: bool = True
    roles: dict[RoleName, RoleConfig]

    @model_validator(mode="after")
    def ensure_required_roles(self) -> "AgentsConfig":
        required: set[RoleName] = {"diagnostic", "treatment", "safety", "monitoring", "coordinator"}
        missing = required.difference(self.roles)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing agent role configuration: {missing_list}")
        for role in SPECIALIST_ROLES:
            if self.roles[role].weight is None:
                raise ValueError(f"Agent role '{role}' must define a positive weight.")
        if self.roles["coordinator"].weight is not None:
            raise ValueError("Coordinator role must not define a weight; only specialist agents are weighted.")
        return self


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_llm_judge: bool = False
    enable_bertscore: bool = False
    judge_llm: str | None = None
    bertscore_model_type: str = "distilbert-base-uncased"
    bertscore_lang: str = "en"


class EvolutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_weight_delta: float = 0.15
    weight_step: float = 0.05
    max_history: int = 200


class RetrievalConfig(BaseModel):
    """Paper defaults: hybrid retrieval weighted 0.8 semantic, 0.2 keyword."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    kb_dir: str = "data/kb"
    embedding_model: str = "BAAI/bge-m3"
    dense_weight: float = 0.8
    sparse_weight: float = 0.2
    top_k: int = 5
    max_top_k: int = 9


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_registry: dict[str, LLMConfig]
    runtime: RuntimeConfig
    agents: AgentsConfig
    evaluation: EvaluationConfig
    evolution: EvolutionConfig
    logging: LoggingConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path)
        load_env_file(config_path.resolve().parent / ".env")
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls(
            llm_registry=raw.get("llm", {}),
            runtime=raw.get("runtime", {}),
            agents=raw.get("agents", {}),
            evaluation=raw.get("evaluation", {}),
            evolution=raw.get("evolution", {}),
            logging=raw.get("logging", {}),
            retrieval=raw.get("retrieval", {}),
        )

    def llm_for(self, key: str) -> LLMConfig:
        if key not in self.llm_registry:
            raise KeyError(f"Unknown LLM config key: {key}")
        return self.llm_registry[key]

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.runtime.artifacts_dir)

    @property
    def evolution_state_path(self) -> Path:
        return Path(self.runtime.evolution_state_path)


def configure_logging(config: AppConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
