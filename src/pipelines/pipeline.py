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
from src.utils.helpers import collate_fn_span, parse_config, set_device, set_seeds
from src.models.binary_align_trainer import BinaryAlignTrainer


@dataclass
class AlignmentGenerationPipeline(LoggedProcess):
    tokenizer: Optional[PreTrainedTokenizer]
    task: str = "all"
    device: str = "auto"
    is_model_deterministic: bool = True
    seed: Optional[int] = 42
    train_dataset_config: Optional[DatasetConfig] = None
    val_dataset_config: Optional[DatasetConfig] = None
    test_dataset_config: Optional[DatasetConfig] = None
    train_data: Optional[AlignmentDatasetSilver] = field(init=False)
    val_data: Optional[AlignmentDatasetSilver] = field(init=False)
    test_data: Optional[AlignmentDatasetSilver] = field(init=False)
    trainer: Optional[BinaryAlignTrainer] = field(init=False)

    def __post_init__(self):
        logger.info("Initialising logger...")
        try:
            LoggedProcess.__init__(self, self.dataset_config.log_output_dir)  # type: ignore
            logger.success("Logger initialised.")
        except KeyError:
            logger.warning("Unable to find variable log_output_dir in configuration.")
            logger.warning("Proceeding with a logger without customization.")

        self._initalise_tokenizer()
        self.device = set_device(device_type=self.device)
        self.seed = set_seeds(
            seed_num=self.seed, deterministic=self.is_model_deterministic
        )

        logger.success(f"{self.__class__.__name__} initialized successfully")

    @logger.catch(message="Unable to complete all-task execution", reraise=True)
    def run_all(self, model_config: ModelConfig, train_config: TrainConfig):
        self.run_data_preparation()
        self.run_training(model_config=model_config, train_config=train_config)
        self.run_prediction()

    @logger.catch(message="Unable to complete pipeline execution.", reraise=True)
    def run_data_preparation(self):
        train_dataset, val_dataset, test_dataset = None, None, None
        if self.task == "all" or self.task == "data":
            if self.train_dataset_config:
                train_dataset = (
                    AlignmentDatasetSilver(
                        tokenizer=self.tokenizer,
                        **self.train_dataset_config,  # type: ignore
                    )
                    if self.train_dataset_config
                    else None
                )
            if self.val_dataset_config:
                val_dataset = (
                    AlignmentDatasetSilver(
                        tokenizer=self.tokenizer,
                        **self.val_dataset_config,  # type: ignore
                    )
                    if self.val_dataset_config
                    else None
                )
            if self.test_dataset_config:
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
                f"{self.__class__.__name__} train_data, val_data and test_data have been updated.\
                Please inspect these variables to ensure they have been prepared properly."
            )

    @logger.catch(message="Unable to complete model training.", reraise=True)
    def run_training(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ):
        # validation checks
        self._validate_training_requirements()
        self.trainer = BinaryAlignTrainer(
            tokenizer=self.tokenizer,  # type: ignore
            model_config=model_config,
            train_config=train_config,
            dataset_config=self.train_dataset_config,  # type: ignore
            dataloader_config=dataloader_config,
            train_data=self.train_data,  # type: ignore
            eval_data=self.val_data,
            device_type=self.device,
            seed_num=self.seed,  # type: ignore
        )

        # update the instance model
        self.trained_model = self.trainer.model

    @logger.catch(message="Unable to complete model prediction.", reraise=True)
    def run_prediction(self):
        if not self.task == "all" or not self.task == "predict":
            logger.error(f"{self.__class__.__name__} not configured for prediction.")

        raise NotImplementedError

    @logger.catch(message="Unable to validate training requirements.", reraise=True)
    def _validate_training_requirements(self) -> bool:
        # returns boolean to allow developer to debug easily if needed
        if not self.task == "all" or not self.task == "train":
            logger.error(f"{self.__class__.__name__} not configured for training.")
            return False

        if self.train_data is None or self.train_dataset_config is None:
            logger.error(
                "Unable to train without training data or a train_dataset_config. \
                Please load some training data in the form of an AlignmentDatasetSilver class \
                to continue with training. To save time and effort, you should prepare the \
                AlignmentDatasetSilver instance with save=True to prevent uncessary \
                re-preparation of datasets."
            )
            return False

        if self.tokenizer is None:
            logger.error("Unable to train without a valid tokenizer.")
            return False

        if self.trained_model is not None:
            logger.warning(
                "A trained model already exists for this run. It will be replaced after training."
            )

        return True

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
    val_dataset_config = parse_config(
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
    val_data = AlignmentDatasetSilver(tokenizer=tok, **val_dataset_config)
