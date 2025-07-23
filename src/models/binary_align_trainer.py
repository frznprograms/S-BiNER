import os
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Union

import torch
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
from src.datasets.datasets_gold import AlignmentDatasetGold
from src.datasets.datasets_silver import AlignmentDatasetSilver
from src.models.binary_align_eval import BinaryAlignEvaluator
from src.models.binary_align_factory import BinaryTokenClassificationFactory
from src.utils.decorators import timed_execution
from src.utils.helpers import collate_fn_span, set_device, set_seeds, init_wandb_tracker


@dataclass
class BinaryAlignTrainer:
    tokenizer: PreTrainedTokenizer
    model_config: ModelConfig
    train_config: TrainConfig
    dataset_config: DatasetConfig
    dataloader_config: DataLoaderConfig
    train_data: Union[AlignmentDatasetGold, AlignmentDatasetSilver]
    eval_data: Optional[Union[AlignmentDatasetGold, AlignmentDatasetSilver]] = None
    device_type: str = "auto"
    seed_num: int = 42
    checkpoint_dir: str = "checkpoints"
    debug_mode: bool = False
    project_name: str = "binary-align-for-zh-ner"

    def __post_init__(self):
        logger.debug("Initialising BinaryAlignTrainer...")
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
        logger.debug("Starting training...")

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
        global_step, global_step_last_logged = 0, 0
        total_loss_scalar = 0.0
        lowest_recorded_loss = float("inf")
        num_steps_testing_patience = 0

        for epoch in range(self.train_config.num_train_epochs):
            for batch in self.train_dataloader:
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
                        total_loss_scalar / (global_step - global_step_last_logged), 4
                    )
                    global_step_last_logged = global_step
                    total_loss_scalar = 0.0
                    if self.accelerator.is_main_process:
                        self.accelerator.log(
                            {
                                "epoch": epoch,
                                "train_loss": tr_loss,
                                "learning_rate": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )
                    # Use pbar.write instead of print to avoid interrupting progress bar
                    pbar.write(f"Batch Training loss: {tr_loss}")
                    self._save_model_checkpoint(pbar=pbar, global_step=global_step)
                    self._cleanup_old_checkpoints(pbar=pbar)

                pbar.update(1)

            # Evaluation
            if self.eval_data is not None and hasattr(self, "eval_dataloader"):
                try:
                    self.accelerator.wait_for_everyone()
                    pbar.write("Evaluating...")
                    metrics = self.evaluate()
                    if metrics is not None:
                        precision, recall, aer, f1 = metrics
                        eval_msg = f"Step {global_step + 1} | Precision: {precision:.4f}, Recall: {recall:.4f}, AER: {aer:.4f}, F1: {f1:.4f}"
                        pbar.write(eval_msg)

                        if self.accelerator.is_main_process:
                            self.accelerator.log(
                                {
                                    "precision": precision,
                                    "recall": recall,
                                    "aer": aer,
                                    "f1": f1,
                                },
                                step=global_step,
                            )
                        # test patience; stop training if loss does not improve over accepted number of steps
                        if train_config.early_stopping_patience:
                            if f1 < lowest_recorded_loss:
                                lowest_recorded_loss = f1
                            else:
                                num_steps_testing_patience += 1
                                if (
                                    num_steps_testing_patience
                                    >= train_config.early_stopping_patience
                                ):
                                    logger.warning(
                                        f"Stopping training at step {global_step} as validation loss did not improve."
                                    )
                                    break

                except Exception as e:
                    pbar.write(f"Evaluation failed: {e}, continuing training...")

        # save model state from end of training
        self.accelerator.save_state(self.checkpoint_dir / "checkpoint-final-checkpoint")  # type: ignore

        pbar.close()
        logger.success("Training completed successfully.")

        # End of training cleanup
        self.accelerator.end_training()
        self.accelerator.free_memory()

    @logger.catch(message="Failed to perform evaluation.", reraise=True)
    def evaluate(self):
        if not self.evaluator:
            self.evaluator = BinaryAlignEvaluator()

        # Use default or configured values
        threshold = getattr(self.model_config, "threshold", 0.7)
        combine_type = getattr(self.model_config, "bidirectional_combine_type", "union")
        tk2word_prob = getattr(self.model_config, "tk2word_prob", "max")

        # Sure alignments only
        sure = self.eval_data.alignment_sure  # type:ignore expected: list[set[tuple[int, int]]]

        return self.evaluator.run(
            dataloader=self.eval_dataloader,  # type: ignore
            model=self.model,  # type: ignore
            threshold=threshold,
            sure=sure,
            device=self.accelerator.device,  # Use accelerator device consistently
            mini_batch_size=self.model_config.batch_size // 2,
            bidirectional_combine_type=combine_type,
            tk2word_prob=tk2word_prob,
        )

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
            self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore
            logger.success("Special tokens added.")

        # Prepare Train Dataloader FIRST (before calculating steps)
        collator = partial(collate_fn_span, tokenizer=self.tokenizer)
        self.dataloader_config.collate_fn = collator
        self.train_dataloader = DataLoader(
            self.train_data.data,  # type:ignore
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


# TODO: add the ability to load from last checkpoint in the event of training failure, along with the data that has yet to be seen by the model
# TODO: consider wandb integration
# TODO (future): dockerize?

if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_config = ModelConfig(model_name_or_path="FacebookAI/roberta-base")
    train_config = TrainConfig(experiment_name="trainer-test", mixed_precision="no")
    train_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/train.src",
        target_lines_path="data/cleaned_data/train.tgt",
        alignments_path="data/cleaned_data/train.talp",
        limit=50,
    )
    eval_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/dev.src",
        target_lines_path="data/cleaned_data/dev.tgt",
        alignments_path="data/cleaned_data/dev.talp",
        limit=10,
        do_inference=True,
    )
    dataloader_config = DataLoaderConfig(collate_fn=collate_fn_span)
    tok = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    train_data = AlignmentDatasetSilver(tokenizer=tok, **train_dataset_config.__dict__)
    eval_data = AlignmentDatasetSilver(tokenizer=tok, **eval_dataset_config.__dict__)

    trainer = BinaryAlignTrainer(
        tokenizer=tok,
        model_config=model_config,
        train_config=train_config,
        dataset_config=train_dataset_config,
        dataloader_config=dataloader_config,
        train_data=train_data,
        eval_data=eval_data,
        seed_num=1,
    )
    trainer.run()
