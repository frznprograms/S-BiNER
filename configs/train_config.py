from abc import ABC
from dataclasses import dataclass
from typing import Optional

# TODO: how can these configs be integrated into training?


@dataclass
class TrainConfig(ABC):
    experiment_name: str
    num_train_epochs: int = 5
    mixed_precision: str = "fp32"
    log_with: Optional[str] = "wandb"
