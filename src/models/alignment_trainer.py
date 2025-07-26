from dataclasses import dataclass
from typing import Optional

from easydict import EasyDict
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.datasets.alignment_pair_dataset import AlignmentPairDataset
from src.models.binary_align_factory import BinaryTokenClassificationFactory
from src.utils.helpers import init_wandb_tracker, set_device, set_seeds


@dataclass
class AlignmentTrainer:
    tokenizer: PreTrainedTokenizer
    model_config: ModelConfig
    train_config: TrainConfig
    dataset_config: DatasetConfig
    dataloader_config: DataLoaderConfig
    train_data: AlignmentPairDataset
    eval_data: Optional[AlignmentPairDataset] = None
    device_type: str = "auto"
    seed_num: int = 42
    checkpoint_dir: str = "checkpoints"
    debug_mode: bool = False
    project_name: str = "binary-align-for-zh-ner"

    def __post_init__(self):
        logger.debug(f"Initialising {self.__class__.__name__}...")
        # Device and seed setup
        self.user_defined_device = set_device(self.device_type)
        self.SEED = set_seeds(self.seed_num)
        logger.debug(f"Set device to {self.user_defined_device}.")
        logger.debug(f"Set seed to {self.SEED}.")

        self.train_config = EasyDict(self.train_config.__dict__)  # type: ignore
        self.model_config = EasyDict(self.model_config.__dict__)  # type: ignore
        self.dataset_config = EasyDict(self.dataset_config.__dict__)  # type: ignore
        self.dataloader_config = EasyDict(self.dataloader_config.__dict__)  # type: ignore
        self.train_dataloader, self.eval_dataloader, self.evaluator = None, None, None

        # setup wandb account if any
        if self.train_config.log_with == "wanb":
            init_wandb_tracker()

        logger.debug("Loaded configuration objects.")

        # Initialize model factory
        self.model_factory = BinaryTokenClassificationFactory(
            model_name_or_path=self.model_config.model_name_or_path,
            config=ModelConfig(**self.model_config),  # type: ignore
        )

        # Initialize model
        self.model = self.model_factory()
        logger.success("Model initialized.")
        logger.success(f"{self.__class__.__name__} initialized successfully")
