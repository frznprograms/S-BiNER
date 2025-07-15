from src.models.binary_align_models import (
    XLMRobertaModelForBinaryTokenClassification,
    RobertaModelForBinaryTokenClassification,
)
from loguru import logger


class BinaryTokenClassificationFactory:
    def __init__(self, model_name_or_path: str, config):
        if "xlm" in model_name_or_path:
            return XLMRobertaModelForBinaryTokenClassification(config=config)
        elif "roberta" in model_name_or_path:
            return RobertaModelForBinaryTokenClassification(config=config)
        else:
            logger.error("Model must be XLM-based or Roberta-based!")
