from abc import ABC
from dataclasses import dataclass
from typing import Optional
from transformers import RobertaConfig, XLMRobertaConfig


@dataclass
class ModelConfig(ABC):
    model_name_or_path: str
    is_pretrained: bool = False
    learning_rate: float = 2e-5
    threshold: float = 0.5
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2
    num_labels: int = 2
    batch_size: int = 32
    classifier_dropout: Optional[float] = None
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 768
    gradient_checkpointing: bool = True
    model_save_path: str = "output"

    # These will be set from the transformer model config when initializing
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    type_vocab_size: Optional[int] = None
    initializer_range: Optional[float] = None
    layer_norm_eps: Optional[float] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        # Set classifier_dropout to hidden_dropout_prob if not provided
        if self.classifier_dropout is None:
            self.classifier_dropout = self.hidden_dropout_prob

    def _to_roberta_config(self) -> RobertaConfig:
        if self.is_pretrained:
            roberta_config = RobertaConfig.from_pretrained(self.model_name_or_path)
        else:
            # Create a new RobertaConfig with sensible defaults
            roberta_config = RobertaConfig(
                vocab_size=self.vocab_size or 50265,
                hidden_size=self.hidden_size,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=self.hidden_dropout_prob,
                max_position_embeddings=self.max_position_embeddings or 514,
                type_vocab_size=self.type_vocab_size or 1,
                initializer_range=self.initializer_range or 0.02,
                layer_norm_eps=self.layer_norm_eps or 1e-5,
                pad_token_id=self.pad_token_id or 1,
                bos_token_id=self.bos_token_id or 0,
                eos_token_id=self.eos_token_id or 2,
            )

        # Override with any custom settings
        roberta_config.hidden_dropout_prob = self.hidden_dropout_prob
        roberta_config.gradient_checkpointing = self.gradient_checkpointing

        return roberta_config  # type:ignore

    def _to_xlm_roberta_config(self) -> XLMRobertaConfig:
        if self.is_pretrained:
            xlm_roberta_config = XLMRobertaConfig.from_pretrained(
                self.model_name_or_path
            )
        else:
            # Create a new XLMRobertaConfig with sensible defaults
            xlm_roberta_config = XLMRobertaConfig(
                vocab_size=self.vocab_size or 50265,
                hidden_size=self.hidden_size,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=self.hidden_dropout_prob,
                max_position_embeddings=self.max_position_embeddings or 514,
                type_vocab_size=self.type_vocab_size or 1,
                initializer_range=self.initializer_range or 0.02,
                layer_norm_eps=self.layer_norm_eps or 1e-5,
                pad_token_id=self.pad_token_id or 1,
                bos_token_id=self.bos_token_id or 0,
                eos_token_id=self.eos_token_id or 2,
            )

        # Override with any custom settings
        xlm_roberta_config.hidden_dropout_prob = self.hidden_dropout_prob
        xlm_roberta_config.gradient_checkpointing = self.gradient_checkpointing

        return xlm_roberta_config  # type:ignore
