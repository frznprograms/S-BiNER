from dataclasses import dataclass
from typing import Optional, Union

import torch
import yaml
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer

from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from src.models.binary_align_factory import BinaryTokenClassificationFactory
from src.utils.decorators import timed_execution
from src.utils.helpers import set_device, set_seeds
from src.utils.pipeline_step import PipelineStep


@dataclass
class BinaryAlignTrainer(PipelineStep):
    tokenizer: PreTrainedTokenizer
    model_config_yaml_or_obj: Union[ModelConfig, str]
    train_config_yaml_or_obj: Union[TrainConfig, str]
    train_dataloader: DataLoader
    context_sep: str = " [WORD_SEP] "
    device_type: str = "auto"
    seed_num: Optional[int] = 42

    def __post_init__(self):
        logger.info("Initialising BinaryAlignTrainer...")

        # Device and seed setup
        self.user_defined_device = set_device(self.device_type)
        self.SEED = set_seeds(self.seed_num)
        logger.success(f"Set device to {self.user_defined_device}.")
        logger.success(f"Set seed to {self.SEED}.")

        # Parse configurations
        self.train_config = self._parse_configs(
            self.train_config_yaml_or_obj, TrainConfig
        )
        self.model_config = self._parse_configs(
            self.model_config_yaml_or_obj, ModelConfig
        )
        logger.success("Loaded configuration files.")

        # Initialize model
        self.model = BinaryTokenClassificationFactory(
            model_name_or_path=self.model_config.model_name_or_path,
            config=self.model_config,
        )
        logger.success("Model initialized.")
        logger.success("BinaryAlignTrainer initialised.")

    @logger.catch(message="Failed to parse configs.", reraise=True)
    def _parse_configs(
        self,
        config_yaml_or_obj: Union[ModelConfig, TrainConfig, str],
        config_class: type,
    ) -> Union[ModelConfig, TrainConfig, None]:
        if isinstance(config_yaml_or_obj, (ModelConfig, TrainConfig)):
            return config_yaml_or_obj
        elif isinstance(config_yaml_or_obj, str):
            if config_yaml_or_obj.endswith((".yaml", ".yml")):
                with open(config_yaml_or_obj, "r") as f:
                    loaded_configs = yaml.safe_load(f)
            else:
                loaded_configs = yaml.safe_load(config_yaml_or_obj)

            # Convert dict to config object
            if config_class == ModelConfig:
                return ModelConfig(**loaded_configs)
            elif config_class == TrainConfig:
                return TrainConfig(**loaded_configs)
            else:
                logger.error(f"Unknown config class: {config_class}")
        else:
            logger.error(f"Invalid config type: {type(config_yaml_or_obj)}")

    @timed_execution
    @logger.catch(message="Failed to complete training.", reraise=True)
    def run(self):
        logger.info("Starting training process...")
        logger.info("Initialising accelerator...")
        self.accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,  # type:ignore
            log_with=self.train_config.log_with,  # type:ignore
        )

        training_device = self.accelerator.device
        if training_device != self.user_defined_device:
            logger.warning(
                f"Mismatch in training devices: \n"
                f"Accelerator device: {training_device}\n"
                f"User specified device: {self.user_defined_device}.\n"
                f"Accelerator device will be used."
            )

        # Initialize tracking
        self.accelerator.init_trackers(project_name=self.train_config.experiment_name)  # type: ignore
        logger.success("Accelerator initialised.")

        # Check if model is pretrained
        if not self.model_config.is_pretrained:  # type:ignore
            logger.info("Adding special context token...")
            special_tokens_dict = {"additional_special_tokens": self.context_sep}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

        number_of_steps = (
            len(self.train_dataloader) * self.train_config.num_train_epochs
        )
        warmup_steps = self.model_config.warmup_ratio * number_of_steps

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (not (any(nd in n for nd in no_decay)))
                ],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (any(nd in n for nd in no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.adamw(
            optimizer_grouped_parameters, lr=self.model.learning_rate
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=number_of_steps
        )

        model, optimizer, scheduler, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, scheduler, self.train_dataloader
        )

        logger.info("Training completed successfully.")
        # End of training cleanup
        self.accelerator.end_training()
        self.accelerator.free_memory()

    @logger.catch(message="Failed to complete evaluation.", reraise=True)
    def eval(self):
        """Execute the evaluation process."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Cannot perform evaluation.")

        logger.info("Starting evaluation process...")

        # TODO: Add your evaluation logic here
        # This is where you'd implement:
        # - Data loading for evaluation
        # - Evaluation loop
        # - Metrics calculation
        # - Results logging

        logger.info("Evaluation completed successfully.")


# TODO: change model configuration from json to a class - leave only the training arguments as a yaml for experimentation
# TODO: consider wandb integration
# TODO (future): dockerize?
