import os
from dataclasses import dataclass, field
from typing import Optional, Union

import yaml
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from src.configs.dataset_config import DatasetConfig
from src.configs.logger_config import LoggedProcess
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.datasets.datasets_gold import AlignmentDatasetGold
from src.datasets.datasets_silver import AlignmentDatasetSilver
from src.models.train import BinaryAlignTrainer
from easydict import EasyDict


@dataclass
class AlignmentGenerationPipeline(LoggedProcess):
    model_config: Union[ModelConfig, str, dict]
    train_config: Union[TrainConfig, str, dict]
    dataset_config: Union[DatasetConfig, str, dict]
    tokenizer: Optional[PreTrainedTokenizer] = None
    log_output_dir: str = "logs"

    # Internal attributes
    _model_config: ModelConfig = field(init=False)
    _train_config: TrainConfig = field(init=False)
    _dataset_config: DatasetConfig = field(init=False)
    _tokenizer: PreTrainedTokenizer = field(init=False)
    _dataset: Union[AlignmentDatasetSilver, AlignmentDatasetGold] = field(init=False)
    _train_dataloader: DataLoader = field(init=False)
    _eval_dataloader: Optional[DataLoader] = field(init=False, default=None)

    def __post_init__(self):
        LoggedProcess.__init__(self, output_dir=self.log_output_dir)

        # Parse all configurations
        self._model_config = self._parse_config(self.model_config, ModelConfig)  # type:ignore
        self._train_config = self._parse_config(self.train_config, TrainConfig)  # type:ignore
        self._dataset_config = self._parse_config(self.dataset_config, DatasetConfig)  # type:ignore
        logger.success("Loaded configurations.")

        self._initialize_tokenizer()
        self._initialize_dataset()
        # TODO: split into train, eval, test

        logger.success(f"{self.__class__.__name__} initialized successfully")

    @logger.catch(message="Unable to parse configs.", reraise=True)
    def _parse_config(self, config: Union[object, str, dict], config_class: type):
        if isinstance(config, config_class):
            return EasyDict(config)
        elif isinstance(config, dict):
            return EasyDict(config_class(**config))
        elif isinstance(config, str):
            # It's a file path
            if os.path.exists(config):
                with open(config, "r") as f:
                    config_dict = yaml.safe_load(f)
                return EasyDict(config_class(**config_dict))
            else:
                # It's a YAML string
                config_dict = yaml.safe_load(config)
                return EasyDict(config_class(**config_dict))
        else:
            logger.error(f"Invalid config type: {type(config)}")

    @logger.catch(message="Unable to initialise tokenizer.", reraise=True)
    def _initialize_tokenizer(self):
        if self.tokenizer is not None:
            self._tokenizer = self.tokenizer
        else:
            logger.info(
                f"Loading tokenizer from {self._model_config.model_name_or_path}"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_config.model_name_or_path
            )

        logger.success("Tokenizer initialized successfully")

    def _initialize_dataset(self):
        dataset_class = self._get_dataset_class(self._dataset_config.data_type)

        self._dataset = dataset_class(
            tokenizer=self._tokenizer,
            source_lines_path=self._dataset_config.source_lines_path,
            target_lines_path=self._dataset_config.target_lines_path,
            alignments_path=self._dataset_config.alignments_path,
            limit=self._dataset_config.limit,
            one_indexed=self._dataset_config.one_indexed,
            context_sep=self._dataset_config.context_sep,
            do_inference=self._dataset_config.do_inference,
            log_output_dir=self._dataset_config.log_output_dir,
            save=self._dataset_config.save,
        )

        logger.success(
            f"Dataset ({self._dataset_config.data_type}) initialized successfully"
        )

    @logger.catch(message="Unable to get dataset class.", reraise=True)
    def _get_dataset_class(self, data_type: str):
        if data_type.lower() == "silver":
            return AlignmentDatasetSilver
        elif data_type.lower() == "gold":
            return AlignmentDatasetGold
        else:
            raise ValueError(
                f"Invalid dataset type: {data_type}. Must be 'silver' or 'gold'"
            )

    def run_training(self):
        logger.info("Starting training pipeline...")

        # Create trainer
        trainer = BinaryAlignTrainer(
            tokenizer=self._tokenizer,
            model_config=self._model_config,
            train_config=self._train_config,
            dataset_config=self._dataset_config,
            train_data=self._train_dataloader,
            eval_data=self._eval_dataloader,
        )

        # Run training
        trainer.run()

        logger.success("Training pipeline completed successfully")
        return trainer

    def run(self):
        logger.info("Starting complete alignment generation pipeline...")
        # Run training
        trainer = self.run_training()

        # Run evaluation (if needed)
        # inference_dataloader = self.run_evaluation()

        logger.success("Complete pipeline executed successfully")
        return trainer
