from dataclasses import dataclass
from typing import Union

from loguru import logger
from transformers import AutoConfig

from configs.model_config import ModelConfig
from src.models.binary_align_models import (
    RobertaModelForBinaryTokenClassification,
    XLMRobertaModelForBinaryTokenClassification,
)


@dataclass
class BinaryTokenClassificationFactory:
    model_name_or_path: str
    config: ModelConfig

    def __call__(
        self,
    ) -> Union[
        XLMRobertaModelForBinaryTokenClassification,
        RobertaModelForBinaryTokenClassification,
        None,
    ]:
        transformer_config = AutoConfig.from_pretrained(self.model_name_or_path)

        # Update config with transformer-specific attributes
        self.config.vocab_size = transformer_config.vocab_size
        self.config.hidden_size = transformer_config.hidden_size
        self.config.max_position_embeddings = getattr(
            transformer_config, "max_position_embeddings", 512
        )
        self.config.type_vocab_size = getattr(transformer_config, "type_vocab_size", 2)
        self.config.initializer_range = getattr(
            transformer_config, "initializer_range", 0.02
        )
        self.config.layer_norm_eps = getattr(
            transformer_config, "layer_norm_eps", 1e-12
        )
        self.config.pad_token_id = getattr(transformer_config, "pad_token_id", 1)
        self.config.bos_token_id = getattr(transformer_config, "bos_token_id", 0)
        self.config.eos_token_id = getattr(transformer_config, "eos_token_id", 2)

        # Set hidden_dropout_prob
        if not hasattr(self.config, "hidden_dropout_prob"):
            self.config.hidden_dropout_prob = getattr(
                transformer_config, "hidden_dropout_prob", 0.1
            )

        if "xlm" in self.model_name_or_path.lower():
            logger.info(f"Creating XLM-RoBERTa model for {self.model_name_or_path}")
            return XLMRobertaModelForBinaryTokenClassification(config=self.config)
        elif "roberta" in self.model_name_or_path.lower():
            logger.info(f"Creating RoBERTa model for {self.model_name_or_path}")
            return RobertaModelForBinaryTokenClassification(config=self.config)
        else:
            logger.error(
                f"Model {self.model_name_or_path} must be XLM-based or RoBERTa-based!"
            )
