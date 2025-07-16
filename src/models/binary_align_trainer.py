from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import torch
from accelerate import Accelerator
from easydict import EasyDict
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer

from src.configs.dataset_config import DatasetConfig
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.datasets.datasets_gold import AlignmentDatasetGold
from src.datasets.datasets_silver import AlignmentDatasetSilver
from src.models.binary_align_factory import BinaryTokenClassificationFactory
from src.utils.decorators import timed_execution
from src.utils.helpers import collate_fn_span, set_device, set_seeds
from src.utils.pipeline_step import PipelineStep


@dataclass
class BinaryAlignTrainer(PipelineStep):
    tokenizer: PreTrainedTokenizer
    model_config: ModelConfig
    train_config: TrainConfig
    dataset_config: DatasetConfig
    train_data: Union[AlignmentDatasetGold, AlignmentDatasetSilver]
    eval_data: Optional[Union[AlignmentDatasetGold, AlignmentDatasetSilver]] = None
    device_type: str = "auto"
    seed_num: Optional[int] = 42

    def __post_init__(self):
        logger.info("Initialising BinaryAlignTrainer...")
        # Device and seed setup
        self.user_defined_device = set_device(self.device_type)
        self.SEED = set_seeds(self.seed_num)
        logger.success(f"Set device to {self.user_defined_device}.")
        logger.success(f"Set seed to {self.SEED}.")

        self.train_config = EasyDict(self.train_config.__dict__)  # type: ignore
        self.model_config = EasyDict(self.model_config.__dict__)  # type: ignore
        self.dataset_config = EasyDict(self.dataset_config.__dict__)  # type: ignore

        logger.success("Loaded configuration objects.")

        # Initialize model factory
        self.model_factory = BinaryTokenClassificationFactory(
            model_name_or_path=self.model_config.model_name_or_path,
            config=ModelConfig(**self.model_config),  # type: ignore
        )

        # Initialize model
        self.model = self.model_factory()
        logger.success("Model initialized.")
        logger.success("BinaryAlignTrainer initialised.")

    @timed_execution
    @logger.catch(message="Failed to complete training.", reraise=True)
    def run(self):
        logger.info("Starting training...")

        logger.info("Initialising accelerator...")
        self.accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,
            log_with=self.train_config.log_with,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
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
        self.accelerator.init_trackers(project_name=self.train_config.experiment_name)
        logger.success("Accelerator initialised.")

        # Check if model is pretrained and handle special tokens
        if not self.model_config.is_pretrained:
            logger.info("Adding special context token...")
            special_tokens_dict = {
                "additional_special_tokens": [self.dataset_config.context_sep]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore
            self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore
            logger.success("Special tokens added.")

        # Calculate training steps
        number_of_steps = len(self.train_data) * self.train_config.num_train_epochs
        warmup_steps = int(self.model_config.warmup_ratio * number_of_steps)

        logger.info("Initialising optimizer...")
        # Setup optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()  # type: ignore
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()  # type: ignore
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.model_config.learning_rate
        )
        logger.success("Optimizer initialised.")

        logger.info("Initialising learning rate scheduler.")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=number_of_steps
        )
        logger.success("Learning rate scheduler initialised.")

        # Prepare Train Dataloader
        collator = partial(collate_fn_span, tokenizer=self.tokenizer)
        self.train_dataloader = DataLoader(
            self.train_data.data,  # type:ignore
            shuffle=True,
            collate_fn=collator,
            batch_size=self.model_config.batch_size,
        )

        # Prepare everything with accelerator
        logger.info("Accelerator preparing training components...")
        self.model, optimizer, scheduler, self.train_dataloader = (
            self.accelerator.prepare(
                self.model, optimizer, scheduler, self.train_dataloader
            )
        )

        # Prepare Eval Dataloader if one is supplied
        if self.eval_data is not None:
            self.eval_dataloader = DataLoader(
                self.eval_data.data,  # type:ignore
                shuffle=True,
                collate_fn=collator,
                batch_size=self.model_config.batch_size,
            )
            self.eval_dataloader = self.accelerator.prepare()
        logger.success("Accelerator prepared training components.")

        # Training loop
        logger.info("Starting training loop...")
        self.accelerator.wait_for_everyone()
        self.accelerator.print(" Num examples = ", len(self.train_data))
        self.accelerator.print(" Num Epochs = ", self.train_config.num_train_epochs)
        self.accelerator.print(
            " Batch Size per device = ", self.model_config.batch_size
        )
        self.accelerator.print(" Total optimization steps = ", number_of_steps)
        pbar = tqdm(
            total=number_of_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        global_step, globalstep_last_logged = 0, 0
        total_loss_scalar = 0.0
        best_aer = 100

        for epoch in range(self.train_config.num_train_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.model.model.device) for k, v in batch.items()}

                loss = self.model(**batch).loss

                self.accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if not self.accelerator.optimizer_step_was_skipped:
                    scheduler.step()

                global_step += 1
                total_loss_scalar += round(loss.item(), 4)

                # Logging
                if global_step % self.train_config.logging_steps == 0:
                    tr_loss = round(
                        total_loss_scalar / (global_step - globalstep_last_logged), 4
                    )
                    globalstep_last_logged = global_step
                    total_loss_scalar = 0.0
                    if self.accelerator.is_main_process:
                        self.accelerator.log(
                            {
                                "train_loss": tr_loss,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "step": global_step,
                            }
                        )
                    self.accelerator.print(tr_loss)

                pbar.update(1)

            # Epoch-level evaluation
            if self.eval_dataloader is not None:
                self.accelerator.wait_for_everyone()
                eval_loss = self.evaluate()
                logger.info(f"Epoch {epoch + 1} evaluation loss: {eval_loss:.4f}")

                if self.accelerator.is_main_process:
                    self.accelerator.log({"eval_loss": eval_loss, "epoch": epoch})

        logger.info("Training completed successfully.")

        # End of training cleanup
        self.accelerator.end_training()
        self.accelerator.free_memory()

    @logger.catch(message="Failed to complete evaluation.", reraise=True)
    def evaluate(self):
        # TODO: evaluator class
        logger.info("Starting evaluation process...")


# TODO: consider wandb integration
# TODO (future): dockerize?
