from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.logger_config import LoggedProcess
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.datasets.datasets_silver import AlignmentDatasetSilver
from src.utils.helpers import collate_fn_span, parse_config


@dataclass
class AlignmentGenerationPipeline(LoggedProcess):
    tokenizer: Optional[PreTrainedTokenizer]
    task: str = "all"
    seed: int = 42
    train_dataset_config: Optional[DatasetConfig] = None
    val_dataset_config: Optional[DatasetConfig] = None
    test_dataset_config: Optional[DatasetConfig] = None
    train_data: Optional[AlignmentDatasetSilver] = field(init=False)
    val_data: Optional[AlignmentDatasetSilver] = field(init=False)
    test_data: Optional[AlignmentDatasetSilver] = field(init=False)

    def __post_init__(self):
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
            train_dataset = (
                AlignmentDatasetSilver(
                    tokenizer=self.tokenizer,
                    **self.train_dataset_config,  # type: ignore
                )
                if self.train_dataset_config
                else None
            )
            val_dataset = (
                AlignmentDatasetSilver(
                    tokenizer=self.tokenizer,
                    **self.val_dataset_config,  # type: ignore
                )
                if self.val_dataset_config
                else None
            )
            test_dataset = (
                AlignmentDatasetSilver(
                    tokenizer=self.tokenizer,
                    **self.test_dataset_config,  # type: ignore
                )
                if self.test_dataset_config
                else None
            )
            self.train_data, self.val_data, self.test_data = (
                train_dataset,
                val_dataset,
                test_dataset,
            )
            logger.info(
                f"{self.__class__.__name__} train_data, val_data and test_data have been updated."
            )
        if self.task == "all" or self.task == "train":
            self._train()
        if self.task == "all" or self.task == "predict":
            self._predict()

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


if __name__ == "__main__":
    model_config = parse_config(
        ModelConfig(model_name_or_path="FacebookAI/roberta-base"),
        config_class=ModelConfig,
    )
    train_config = parse_config(
        TrainConfig(experiment_name="trainer-test", mixed_precision="no"),
        config_class=TrainConfig,
    )
    train_dataset_config = parse_config(
        DatasetConfig(
            source_lines_path="data/cleaned_data/train.src",
            target_lines_path="data/cleaned_data/train.tgt",
            alignments_path="data/cleaned_data/train.talp",
            limit=50,
        ),
        config_class=DatasetConfig,
    )
    eval_dataset_config = parse_config(
        DatasetConfig(
            source_lines_path="data/cleaned_data/dev.src",
            target_lines_path="data/cleaned_data/dev.tgt",
            alignments_path="data/cleaned_data/dev.talp",
            limit=10,
            do_inference=True,
        ),
        config_class=DatasetConfig,
    )
    dataloader_config = DataLoaderConfig(collate_fn=collate_fn_span)
    tok = AutoTokenizer.from_pretrained(model_config.model_name_or_path)  # type: ignore
    train_data = AlignmentDatasetSilver(tokenizer=tok, **train_dataset_config)
    eval_data = AlignmentDatasetSilver(tokenizer=tok, **eval_dataset_config)
