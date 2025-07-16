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

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
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
    dataloader_config: DataLoaderConfig
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
        self.dataloader_config = EasyDict(self.dataloader_config.__dict__)  # type: ignore
        self.train_dataloader, self.eval_dataloader = None, None

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

        # Prepare Train Dataloader FIRST (before calculating steps)
        collator = partial(collate_fn_span, tokenizer=self.tokenizer)
        self.dataloader_config.collate_fn = collator
        self.train_dataloader = DataLoader(
            self.train_data.data,  # type:ignore
            **self.dataloader_config,  # type:ignore
        )

        # NOW calculate training steps using the actual dataloader length
        number_of_steps = (
            len(self.train_dataloader) * self.train_config.num_train_epochs
        )
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
                **self.dataloader_config,  # type:ignore
            )
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        logger.success("Accelerator prepared training components.")

        # Training loop
        logger.info("Starting training loop...")
        self.accelerator.wait_for_everyone()
        self.accelerator.print("*" * 50)
        self.accelerator.print(" Num examples = ", len(self.train_data))
        self.accelerator.print(" Num Epochs = ", self.train_config.num_train_epochs)
        self.accelerator.print(
            " Batch Size per device = ", self.model_config.batch_size
        )
        self.accelerator.print(
            " Total batches per epoch = ", len(self.train_dataloader)
        )
        self.accelerator.print(" Total optimization steps = ", number_of_steps)
        self.accelerator.print("*" * 50)

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
                    self.accelerator.print(f"Batch Training loss: {tr_loss}")

                pbar.update(1)

            # Epoch-level evaluation
            if self.eval_data is not None and hasattr(self, "eval_dataloader"):
                try:
                    self.accelerator.wait_for_everyone()
                    eval_loss = self.evaluate()
                    if eval_loss is not None:
                        logger.info(
                            f"Epoch {epoch + 1} evaluation loss: {eval_loss:.4f}"
                        )
                        if self.accelerator.is_main_process:
                            self.accelerator.log(
                                {"eval_loss": eval_loss, "epoch": epoch}
                            )
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")

        logger.info("Training completed successfully.")
        pbar.close()

        # End of training cleanup
        self.accelerator.end_training()
        self.accelerator.free_memory()

    @logger.catch(message="Failed to complete evaluation.", reraise=True)
    def evaluate(self):
        # TODO: evaluator class
        return None


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
        limit=5,
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
