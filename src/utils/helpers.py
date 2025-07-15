import torch
import random
from typing import Optional
from loguru import logger
from src.models.train import BinaryAlignTrainer
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig


def delist_the_list(items: list):
    for i in range(len(items)):
        items[i] = items[i][0]
    return items


def set_device(device_type: str = "auto") -> str:
    device = None
    if device_type == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    else:
        if device_type == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                logger.warning(
                    f"Could not set device to {device_type}, defaulting to cpu instead."
                )
                device = "cpu"
        if device_type == "mps":
            if torch.mps.is_available():
                device = "mps"
            else:
                logger.warning(
                    f"Could not set device to {device_type}, defaulting to cpu instead."
                )
                device = "cpu"
        else:
            device = "cpu"

    if device is None:
        raise ValueError(f"Could not assign device of type {device_type}")

    return device


def set_seeds(seed_num: Optional[int], deterministic: bool = True) -> None:
    if seed_num is None:
        seed_num = 42
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_trainer_from_yaml(
    model_name_or_path: str,
    model_config_path: str,
    train_config_path: str,
    device_type: str = "auto",
    seed_num: Optional[int] = 42,
) -> BinaryAlignTrainer:
    return BinaryAlignTrainer(
        model_name_or_path=model_name_or_path,
        model_config_yaml_or_obj=model_config_path,
        train_config_yaml_or_obj=train_config_path,
        device_type=device_type,
        seed_num=seed_num,
    )


def create_trainer_from_objects(
    model_name_or_path: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device_type: str = "auto",
    seed_num: Optional[int] = 42,
) -> BinaryAlignTrainer:
    return BinaryAlignTrainer(
        model_name_or_path=model_name_or_path,
        model_config_yaml_or_obj=model_config,
        train_config_yaml_or_obj=train_config,
        device_type=device_type,
        seed_num=seed_num,
    )
