from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from easydict import EasyDict
from loguru import logger
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.logger_config import LoggedProcess
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.datasets.datasets_silver import AlignmentDatasetSilver
from src.utils.helpers import parse_config


@dataclass
class AlignmentGenerationPipeline(LoggedProcess):
    tokenizer: Optional[PreTrainedTokenizer]
    task: str = "all"
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    model_config: Union[ModelConfig, str, dict]  # type: ignore
    train_config: Union[TrainConfig, str, dict]  # type: ignore
    dataset_config: Union[DatasetConfig, str, dict]  # type: ignore
    dataloader_config: Union[DataLoaderConfig, str, dict]  # type: ignore

    train_data: list[dict[str, torch.Tensor]] = field(default_factory=list, init=False)
    val_data: list[dict[str, torch.Tensor]] = field(default_factory=list, init=False)
    test_data: list[dict[str, torch.Tensor]] = field(default_factory=list, init=False)

    def __post_init__(self):
        logger.info("Loading configurations...")
        self.model_config: EasyDict = parse_config(
            config=self.model_config, config_class=ModelConfig
        )
        self.train_config: EasyDict = parse_config(
            config=self.train_config, config_class=TrainConfig
        )
        self.dataset_config: EasyDict = parse_config(
            config=self.dataset_config, config_class=DatasetConfig
        )
        self.dataloader_config: EasyDict = parse_config(
            config=self.dataloader_config, config_class=DataLoaderConfig
        )
        logger.success("Configuratons loaded.")
        logger.info("Initialising logger...")
        try:
            LoggedProcess.__init__(self, self.dataset_config.log_output_dir)  # type: ignore
            logger.success("Logger initialised.")
        except KeyError:
            logger.warning("Unable to find variable log_output_dir in configuration.")
            logger.warning("Proceeding with a logger without customization.")

        self._initalise_tokenizer()

        logger.success(f"{self.__class__.__name__} initialized successfully")

    @logger.catch(message="Unable to complete pipeline execution.", reraise=True)
    def run(self):
        if self.task == "all" or self.task == "data":
            train_data, val_data, test_data = self._prepare_datasets()
            self.train_data, self.val_data, self.test_data = (
                train_data,
                val_data,
                test_data,
            )
            logger.info(
                f"{self.__class__.__name__} train_data, val_data and test_data have been updated."
            )
        if self.task == "all" or self.task == "train":
            self._train()
        if self.task == "all" or self.task == "predict":
            self._predict()

    @logger.catch(message="Unable to prepare datasets.", reraise=True)
    def _prepare_datasets(self):
        logger.info("Preparing dataset...")
        ads = AlignmentDatasetSilver(tokenizer=self.tokenizer, **self.dataset_config)  # type: ignore
        data = ads.data
        logger.success("Dataset prepared.")
        # symmetrization is already done by the model, so actually we do not need to have reverse data
        # but the option is always there in AlignmentDatasetSilver, if anyone wants it
        logger.info("Splitting data into train-val-test sets...")

        train_idx = int(self.train_ratio * len(data))
        val_idx = int((self.train_ratio + self.val_ratio) * len(data))
        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]

        logger.success("Data split into train-val-test sets.")

        return train_data, val_data, test_data

    @logger.catch(message="Unable to complete model training.", reraise=True)
    def _train(self):
        raise NotImplementedError

    @logger.catch(message="Unable to complete model prediction.", reraise=True)
    def _predict(self):
        raise NotImplementedError

    @logger.catch(message="Unable to initialise tokenizer.", reraise=True)
    def _initalise_tokenizer(self):
        if self.tokenizer is None:
            logger.info(
                f"Loading tokenizer from {self.model_config.model_name_or_path}"  # type: ignore
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name_or_path  # type: ignore
            )
        logger.success("Tokenizer initialised.")
