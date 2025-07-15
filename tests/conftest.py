import pytest
from configs.model_config import ModelConfig


@pytest.fixture
def get_model_config(model_name_or_path: str = "facebook-ai/roberta-base"):
    model_configs = ModelConfig(model_name_or_path=model_name_or_path)
    return model_configs
