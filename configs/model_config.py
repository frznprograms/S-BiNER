from abc import ABC
from dataclasses import dataclass
from typing import Optional

# TODO: how can these configs be integrated into training?


@dataclass
class ModelConfig(ABC):
    model_name_or_path: str
    is_pretrained: bool = False
    learning_rate: float = 2e-5
    batch_size: int = 16
    threshold: float = 0.5
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2
    logging_steps: int = 200
    num_labels: int = 2
    classifier_dropout: float = 0.3
    hidden_size: int = 768
    gradient_checkpointing: bool = True
    save_strategy: str = "epoch"
    mixed_precision: Optional[str] = "fp32"
    model_save_path: str = "output"
