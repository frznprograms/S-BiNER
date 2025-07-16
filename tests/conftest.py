from typing import Any, Union

import pytest
from easydict import EasyDict

from src.configs.dataset_config import DatasetConfig
from src.configs.logger_config import LoggedProcess
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig


@pytest.fixture
def get_model_config(
    model_name_or_path: str = "facebook-ai/roberta-base",
    return_as_dict: bool = True,
    **other_configs: Any,
) -> Union[ModelConfig, EasyDict]:
    config_params = {"model_name_or_path": model_name_or_path}
    config_params.update(other_configs)

    model_configs = ModelConfig(**config_params)  # type:ignore
    if return_as_dict:
        model_configs = EasyDict(model_configs.__dict__)
    return model_configs


@pytest.fixture
def get_train_config(
    experiment_name: str = "pytest",
    return_as_dict: bool = True,
    **other_configs: Any,
) -> Union[TrainConfig, EasyDict]:
    config_params = {"experiment_name": experiment_name}
    config_params.update(other_configs)

    train_configs = TrainConfig(**config_params)  # type:ignore
    if return_as_dict:
        train_configs = EasyDict(train_configs.__dict__)
    return train_configs


@pytest.fixture
def get_dataset_config(
    source_lines_path: str = "../data/cleaned_data/train.src",
    target_lines_path: str = "../data/cleaned_data/train.tgt",
    alignments_path: str = "../data/cleaned_data/train.talp",
    limit: int = 10,
    return_as_dict: bool = True,
    **other_configs: Any,
) -> Union[DatasetConfig, EasyDict]:
    config_params = {
        "source_lines_path": source_lines_path,
        "target_lines_path": target_lines_path,
        "alignments_path": alignments_path,
        "limit": limit,
    }
    config_params.update(other_configs)

    dataset_configs = DatasetConfig(**config_params)
    if return_as_dict:
        dataset_configs = EasyDict(dataset_configs.__dict__)
    return dataset_configs


@pytest.fixture
def get_logger(output_dir: str = "./test_logs"):
    log_process = LoggedProcess(output_dir=output_dir)
    return log_process


if __name__ == "__main__":
    try:
        model_configs = get_model_config()
        train_configs = get_train_config(experiment_name="functionality_test")
        dataset_configs = get_dataset_config()
        print(f"ModelConfig keys: {model_configs.keys()} \n")  # type:ignore
        print(f"TrainConfig keys: {train_configs.keys()} \n")  # type:ignore
        print(f"DatasetConfig keys: {dataset_configs.keys()} \n")  # type:ignore

    except Exception as e:
        print(f"Error in obtaining data: {e}")
