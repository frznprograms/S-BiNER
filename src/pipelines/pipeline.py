from dataclasses import dataclass, field
from typing import Optional, Union

from loguru import logger
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.logger_config import LoggedProcess
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.utils.helpers import parse_config


@dataclass
class AlignmentGenerationPipeline(LoggedProcess):
    tokenizer: Optional[PreTrainedTokenizer]
    model_config: Union[ModelConfig, str, dict] = field(init=False)
    train_config: Union[TrainConfig, str, dict] = field(init=False)
    dataset_config: Union[DatasetConfig, str, dict] = field(init=False)
    dataloader_config: Union[DataLoaderConfig, str, dict] = field(init=False)

    def __post_init__(self):
        logger.info("Loading configurations...")
        self.model_config = parse_config(
            config=self.model_config, config_class=ModelConfig
        )
        self.train_config = parse_config(
            config=self.train_config, config_class=TrainConfig
        )
        self.dataset_config = parse_config(
            config=self.dataset_config, config_class=DatasetConfig
        )
        self.dataloader_config = parse_config(
            config=self.dataloader_config, config_class=DataLoaderConfig
        )
        logger.success("Configuratons loaded.")
        logger.info("Initialising logger...")
        try:
            LoggedProcess.__init__(self, self.dataset_config.log_output_dir)  # type: ignore
            logger.success("Logger initialised.")
        except KeyError:
            logger.error("Unable to find variable log_output_dir in configuration.")

        logger.success(f"{self.__class__.__name__} initialized successfully")

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


# TODO: split into train, eval, test for training/testing
