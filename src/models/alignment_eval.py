from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from easydict import EasyDict
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.model_config import ModelConfig
from src.datasets.alignment_pair_dataset import AlignmentPairDataset
from src.models.binary_token_classification import (
    BinaryTokenClassificationModel,
    create_collate_fn,
)
from src.utils.helpers import set_device

# TODO: find out if you need the classes to be frozen. If so, changed them
# such that properties can be modified with setter methods


@dataclass
class AlignmentPairEvaluator:
    checkpoint_path: str
    checkpoint_number: Optional[Union[int, str]]
    eval_data: AlignmentPairDataset
    eval_dataloader_config: DataLoaderConfig
    device_type: str = "cpu"
    debug_mode: bool = False
    project_name: str = "pair-alignment-for-zh-ner"
    threshold: float = 0.5
    pos_weight: int = 15

    def __post_init__(self):
        if self.checkpoint_number is None:
            self.checkpoint_number = "final"  # use the last checkpoint

        self.user_defined_device = set_device(self.device_type)

        map_location = self.user_defined_device
        self.model = BinaryTokenClassificationModel.from_pretrained(
            load_dir=f"{self.checkpoint_path}/checkpoint-{self.checkpoint_number}",
            map_location=map_location,
            strict=True,
        )

        self.eval_dataloader_config = EasyDict(self.eval_dataloader_config.__dict__)  # type: ignore
        self.eval_dataloader = DataLoader(
            self.eval_data.data,
            **self.eval_dataloader_config,  # type: ignore
        )
        self.criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor([self.pos_weight])
        ).to(
            self.user_defined_device
        )  # Element-wise loss

    @torch.no_grad
    @logger.catch(message="Unable to complete evaluation", reraise=True)
    def run(self):
        logger.info("Starting evaluations now...")
        self.model.eval()

        n = len(self.eval_dataloader)
        pbar = tqdm(total=n)

        total_loss, total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0
        for step, batch in enumerate(self.eval_dataloader):  # type: ignore
            input_ids = batch["input_ids"].to(self.user_defined_device)
            attn_mask = batch["attention_mask"].to(self.user_defined_device)
            src_word_ids = batch["source_word_ids"].to(self.user_defined_device)
            tgt_word_ids = batch["target_word_ids"].to(self.user_defined_device)
            labels = batch["labels"].to(self.user_defined_device)
            label_mask = batch["label_mask"].to(self.user_defined_device)

            logits = self.model(input_ids, attn_mask, src_word_ids, tgt_word_ids)  # type: ignore
            loss_matrix = self.criterion(logits, labels)  # (B, S, T)
            per_sample_normalized_loss = (loss_matrix * label_mask).sum(dim=(1, 2)) / (
                label_mask.sum(dim=(1, 2)) + 1e-8
            )
            masked_loss = per_sample_normalized_loss.mean()
            precision, recall, f1 = self.calculate_metrics(
                logits, labels, label_mask, threshold=self.threshold
            )

            total_loss += masked_loss.item()
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            pbar.update(1)

        metrics = {
            "BCE Loss": total_loss / n,
            "Precision": total_precision / n,
            "Recall": total_recall / n,
            "F1": total_f1 / n,
        }
        print(f"Average Loss: {total_loss / n}")
        print(f"Average Precision: {total_precision / n}")
        print(f"Average Recall: {total_recall / n}")
        print(f"Average f1_score: {total_f1 / n}")

        logger.success("Evaluations completed.")

        return metrics

    @logger.catch(message="Failed to calculate metrics", reraise=True)
    def calculate_metrics(self, logits, labels, mask, threshold: float = 0.5):
        # logits, labels: (B,S,T) float; mask: (B,S,T) bool
        with torch.no_grad():
            pred = (logits.sigmoid() >= threshold) & mask.bool()
            gold = (labels >= 0.5) & mask.bool()

            tp = (pred & gold).sum().item()
            fp = (pred & ~gold).sum().item()
            fn = (~pred & gold).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1


if __name__ == "__main__":
    eval_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/dev.src",
        target_lines_path="data/cleaned_data/dev.tgt",
        alignments_path="data/cleaned_data/dev.talp",
        limit=500,
        do_inference=True,
    )

    model_config = ModelConfig(model_name_or_path="FacebookAI/roberta-base")

    tok = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, add_prefix_space=True
    )

    eval_dataloader_config = DataLoaderConfig(
        collate_fn=create_collate_fn(tokenizer=tok), shuffle=False
    )

    eval_data = AlignmentPairDataset(
        tokenizer=tok,
        **eval_dataset_config.__dict__,
        dataloader_config=eval_dataloader_config,
    )

    e = AlignmentPairEvaluator(
        checkpoint_path="checkpoints",
        checkpoint_number=300,
        eval_data=eval_data,
        eval_dataloader_config=eval_dataloader_config,
    )
    e.run()
