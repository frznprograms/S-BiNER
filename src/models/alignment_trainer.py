import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from easydict import EasyDict
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.datasets.alignment_pair_dataset import AlignmentPairDataset
from src.models.binary_align_factory import BinaryTokenClassificationFactory
from src.models.binary_token_classification import create_collate_fn
from src.utils.decorators import timed_execution
from src.utils.helpers import (
    init_wandb_tracker,
    set_device,
    set_seeds,
)


@dataclass
class AlignmentTrainer:
    tokenizer: PreTrainedTokenizer
    model_config: ModelConfig
    train_config: TrainConfig
    dataset_config: DatasetConfig
    dataloader_config: DataLoaderConfig
    train_data: AlignmentPairDataset
    eval_data: Optional[AlignmentPairDataset] = None
    device_type: str = "cpu"  # TODO: change back to auto when cuda is ready
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

    @timed_execution
    @logger.catch(message="Failed to complete training.", reraise=True)
    def run(self):
        optimizer, scheduler, number_of_steps = self._init_training_utils()

        # Prepare everything with accelerator
        logger.debug("Accelerator preparing training components...")
        self.model, optimizer, scheduler, self.train_dataloader = (
            self.accelerator.prepare(
                self.model, optimizer, scheduler, self.train_dataloader
            )
        )

        # Prepare Eval Dataloader if one is supplied
        if self.eval_data is not None:
            self.eval_dataloader = DataLoader(
                self.eval_data.data,  # type:ignore
                # TODO: see if can get batched eval to work
                # **self.dataloader_config,  # type:ignore
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                shuffle=True,
            )
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        logger.success("Accelerator prepared training components.")

        # Training loop
        self.accelerator.wait_for_everyone()
        logger.debug("Starting Training...")
        self.accelerator.print("=" * 50)
        self.accelerator.print(" Num examples = ", len(self.train_data))
        self.accelerator.print(" Num Epochs = ", self.train_config.num_train_epochs)
        self.accelerator.print(
            " Batch Size per device = ", self.model_config.batch_size
        )
        self.accelerator.print(" Total optimization steps = ", number_of_steps)
        self.accelerator.print(f" Training device =  {self.accelerator.device}")
        self.accelerator.print("=" * 50)

        pbar = tqdm(
            total=number_of_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")  # Element-wise loss
        # TODO: implement early stopping after eval is set up properly

        self.model.train()
        total_steps = 0
        total_loss_all_epochs = 0.0  # Track loss across all epochs for final return
        for epoch in range(self.train_config.num_train_epochs):
            total_masked_loss = 0.0
            epoch_steps = 0  # Track steps within this epoch
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.user_defined_device)
                attention_mask = batch["attention_mask"].to(self.user_defined_device)
                source_token_to_word_mapping = batch["source_word_ids"].to(
                    self.user_defined_device
                )
                target_token_to_word_mapping = batch["target_word_ids"].to(
                    self.user_defined_device
                )
                labels = batch["labels"].to(self.user_defined_device)
                print(f"Labels shape: {labels.shape}")
                label_mask = batch["label_mask"].to(self.user_defined_device).float()
                print(f"Label mask shape: {label_mask.shape}")

                # Forward pass
                logits = self.model(
                    input_ids,
                    attention_mask,
                    source_token_to_word_mapping,
                    target_token_to_word_mapping,
                )  # (B, S, T)

                # Align label shape with logits
                # B, S_pred, T_pred = logits.shape
                # labels = labels[:, :S_pred, :T_pred]
                # label_mask = label_mask[:, :S_pred, :T_pred]

                assert logits.shape == labels.shape == label_mask.shape, (
                    f"Shape mismatch: logits {logits.shape}, labels {labels.shape}, label_mask {label_mask.shape}"
                )

                # Element-wise loss then mask
                loss_matrix = self.criterion(logits, labels)  # (B, S, T)
                masked_loss = (loss_matrix * label_mask).sum() / label_mask.sum()

                # Backward and optimizer step
                optimizer.zero_grad()
                self.accelerator.backward(masked_loss)
                optimizer.step()
                scheduler.step()

                total_masked_loss += masked_loss.item()
                total_loss_all_epochs += masked_loss.item()
                epoch_steps += 1
                self._save_model_checkpoint(pbar=pbar, global_step=total_steps)
                self._cleanup_old_checkpoints(pbar=pbar)
                pbar.update(1)
                total_steps += 1

            # Calculate average loss for this epoch using epoch_steps
            avg_epoch_loss = total_masked_loss / epoch_steps
            pbar.write(f"Epoch {epoch + 1}: Avg masked loss = {avg_epoch_loss:.6f}")

        # Return average loss across all epochs and all steps
        return total_loss_all_epochs / total_steps

    @torch.no_grad
    @logger.catch(message="Failed to perform evaluation.", reraise=True)
    def evaluate(self):
        # TODO: implement evaluation
        self.model.eval()  # type: ignore
        total_loss = 0.0
        for batch in tqdm(self.eval_dataloader, desc="Running Validation"):
            input_ids = batch["input_ids"].to(self.user_defined_device)
            attention_mask = batch["attention_mask"].to(self.user_defined_device)
            source_token_to_word_mapping = batch["source_word_ids"].to(
                self.user_defined_device
            )
            target_token_to_word_mapping = batch["target_word_ids"].to(
                self.user_defined_device
            )
            labels = batch["labels"].to(self.user_defined_device)
            # TODO: use label mask find masked loss, which ensures padding != wrong alignment
            logits = self.model(
                input_ids,
                attention_mask,
                source_token_to_word_mapping,
                target_token_to_word_mapping,
            )  # type: ignore # (B, S, T)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            return total_loss / len(self.eval_dataloader)  # type: ignore

    @logger.catch(message="Failed to initialise training utils", reraise=True)
    def _init_training_utils(self):
        logger.debug("Initialising accelerator...")

        # Set environment variable to force Accelerator to use the specified device
        if self.user_defined_device == "mps":
            os.environ["ACCELERATE_TORCH_DEVICE"] = "mps"
        elif self.user_defined_device == "cuda":
            os.environ["ACCELERATE_TORCH_DEVICE"] = (
                f"cuda:{self.user_defined_device.index or 0}"
            )
        else:
            os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"

        self.accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,
            log_with=self.train_config.log_with,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
        )

        # Initialize tracking
        self.accelerator.init_trackers(
            project_name=self.train_config.experiment_name,
        )
        logger.success("Accelerator initialised.")

        # Check if model is pretrained and handle special tokens
        if not self.model_config.is_pretrained:
            logger.debug(
                "Since model is not pretrained, adding special context token..."
            )
            special_tokens_dict = {
                "additional_special_tokens": [self.dataset_config.context_sep]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)  # type: ignore
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))  # type: ignore
            logger.success("Special tokens added.")

        # Prepare Train Dataloader FIRST (before calculating steps)
        self.dataloader_config.collate_fn = create_collate_fn(self.tokenizer)
        self.train_dataloader = DataLoader(
            self.train_data.data,  # type: ignore
            **self.dataloader_config,  # type:ignore
        )

        # calculate training steps using the actual dataloader length
        number_of_steps = (
            len(self.train_dataloader) * self.train_config.num_train_epochs
        )
        warmup_steps = int(self.model_config.warmup_ratio * number_of_steps)

        # Setup optimizer
        logger.debug("Initialising optimizer...")
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

        logger.debug("Initialising learning rate scheduler.")
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=number_of_steps
        )
        logger.success("Learning rate scheduler initialised.")

        return optimizer, scheduler, number_of_steps

    @logger.catch(message="Failed to save model checkpoint", reraise=True)
    def _save_model_checkpoint(
        self,
        pbar,
        global_step: int,
    ):
        should_save_checkpoint = False

        if hasattr(self.train_config, "save_strategy"):
            if self.train_config.save_strategy == "steps":
                if (
                    hasattr(self.train_config, "save_steps")
                    and global_step % self.train_config.save_steps == 0  # type: ignore
                ):
                    should_save_checkpoint = True
            elif self.train_config.save_strategy == "epoch":
                # Call at the end of each epoch loop
                should_save_checkpoint = True

        # Default fallback to saving by steps
        elif hasattr(self.train_config, "save_steps"):
            if global_step % self.train_config.save_steps == 0:  # type: ignore
                should_save_checkpoint = True

        if should_save_checkpoint and self.accelerator.is_main_process:
            try:
                # Create the specific checkpoint directory
                checkpoint_path = (
                    Path(self.checkpoint_dir) / f"checkpoint-{global_step}"
                )
                checkpoint_path.mkdir(parents=True, exist_ok=True)

                # Save the model state to the specific checkpoint directory
                self.accelerator.save_state(checkpoint_path)  # type: ignore

                pbar.write(
                    f"Checkpoint saved at step {global_step} to {checkpoint_path}"
                )
                # Clean up old checkpoints if save_total_limit is set
                if (
                    hasattr(self.train_config, "save_total_limit")
                    and self.train_config.save_total_limit > 0  # type: ignore
                ):
                    self._cleanup_old_checkpoints(pbar)

            except Exception as e:
                pbar.write(f"Failed to save checkpoint at step {global_step}.")
                logger.warning(e)

    @logger.catch(message="Failed to delete old checkpoints", reraise=True)
    def _cleanup_old_checkpoints(self, pbar):
        """Needs to take pbar as an argument so that the print statements are
        nicer to read."""
        checkpoint_dirs = []
        for item in Path(self.checkpoint_dir).iterdir():
            if (
                item.is_dir()
                and item.name.startswith("checkpoint-")
                and item.name != "final-checkpoint"
            ):
                try:
                    step_num = int(item.name.split("-")[1])
                    checkpoint_dirs.append((item.name, step_num, item))
                except (IndexError, ValueError):
                    continue

        # Sort by step number (newest first)
        checkpoint_dirs.sort(key=lambda x: x[1], reverse=True)

        # Keep only save_total_limit most recent checkpoints
        if len(checkpoint_dirs) > self.train_config.save_total_limit:  # type: ignore
            checkpoints_to_remove = checkpoint_dirs[
                self.train_config.save_total_limit :
            ]

            for dir_name, step_num, full_path in checkpoints_to_remove:
                try:
                    shutil.rmtree(full_path)
                    pbar.write(f"Removed old checkpoint: {dir_name}")
                except Exception as e:
                    pbar.write(f"Failed to remove checkpoint {dir_name}: {e}")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from src.datasets.alignment_pair_dataset import AlignmentPairDataset

    model_config = ModelConfig(model_name_or_path="FacebookAI/roberta-base")
    train_config = TrainConfig(experiment_name="trainer-test", mixed_precision="no")
    train_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/train.src",
        target_lines_path="data/cleaned_data/train.tgt",
        alignments_path="data/cleaned_data/train.talp",
        limit=2,
        debug_mode=False,
    )
    # eval_dataset_config = DatasetConfig(
    #     source_lines_path="data/cleaned_data/dev.src",
    #     target_lines_path="data/cleaned_data/dev.tgt",
    #     alignments_path="data/cleaned_data/dev.talp",
    #     limit=10,
    #     do_inference=True,
    # )
    tok = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, add_prefix_space=True
    )
    dataloader_config = DataLoaderConfig(collate_fn=create_collate_fn(tokenizer=tok))
    train_data = AlignmentPairDataset(
        tokenizer=tok,
        **train_dataset_config.__dict__,
        dataloader_config=dataloader_config,
    )
    # eval_data = AlignmentPairDataset(tokenizer=tok, **eval_dataset_config.__dict__)

    trainer = AlignmentTrainer(
        tokenizer=tok,
        model_config=model_config,
        train_config=train_config,
        dataset_config=train_dataset_config,
        dataloader_config=dataloader_config,
        train_data=train_data,
        eval_data=None,  # TODO: change when eval is set up
        seed_num=1,
    )
    trainer.run()
