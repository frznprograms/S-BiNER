from dataclasses import dataclass
from src.models.binary_align_models import (
    XLMRobertaModelForBinaryTokenClassification,
    RobertaModelForBinaryTokenClassification,
)
from loguru import logger
from configs.model_config import ModelConfig


@dataclass
class BinaryTokenClassificationFactory:
    model_name_or_path: str
    config: ModelConfig

    def __call__(self):
        if "xlm" in self.model_name_or_path:
            return XLMRobertaModelForBinaryTokenClassification(config=self.config)
        elif "roberta" in self.model_name_or_path:
            return RobertaModelForBinaryTokenClassification(config=self.config)
        else:
            logger.error("Model must be XLM-based or Roberta-based!")
