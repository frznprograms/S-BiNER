from dataclasses import dataclass
from typing import Optional

from easydict import EasyDict


@dataclass
class TrainConfig:
    """
    Class which determines training configurations, i.e. configurations to be used
    during the training loop. Mirrors torch training.
    """

    experiment_name: str
    num_train_epochs: int = 3
    mixed_precision: str = "fp16"
    log_with: Optional[str] = None  # TODO: change back to "wandb" when ready
    logging_steps: int = 100
    save_strategy: str = "steps"
    save_steps: Optional[int] = 100
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: Optional[int] = None
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    resume_from_checkpoint: Optional[str] = None


if __name__ == "__main__":
    train_config = TrainConfig(experiment_name="test_1")
    train_config_dict = EasyDict(train_config.__dict__)
    print(train_config_dict.keys())
    print(train_config.eval_strategy)
