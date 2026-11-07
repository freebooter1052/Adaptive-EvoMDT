from evomdt.config import AppConfig


def test_load_config_contains_required_roles():
    config = AppConfig.load("config.toml")
    assert config.runtime.default_llm in config.llm_registry
    assert (config.retrieval.dense_weight, config.retrieval.sparse_weight) == (0.8, 0.2)
    assert set(config.agents.roles) == {"diagnostic", "treatment", "safety", "monitoring", "coordinator"}
    assert config.agents.roles["coordinator"].weight is None
    assert all(config.agents.roles[role].weight == 1.0 for role in ["diagnostic", "treatment", "safety", "monitoring"])
    assert config.evaluation.bertscore_model_type == "distilbert-base-uncased"
