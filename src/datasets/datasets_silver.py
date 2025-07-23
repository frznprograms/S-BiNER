from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.datasets.base_dataset import BaseDataset
from src.utils.helpers import collate_fn_span, parse_single_alignment


@dataclass
class AlignmentDatasetSilver(BaseDataset):
    def __post_init__(self):
        super().__post_init__()
        logger.success(
            f"{self.__class__.__name__} initialized with type \
            inference set to {self.do_inference}."
        )

    def _prepare_labels(
        self,
        data: list,
        source_bpe2word: torch.Tensor,
        target_bpe2word: torch.Tensor,
        input_ids: torch.Tensor,
        input_id_dict: dict[str, Any],
        alignment: str,
        reverse: bool = False,
    ):
        # Create empty labels
        source_labels = torch.ones_like(input_id_dict["source_input_ids"]) * -100
        target_labels = torch.zeros_like(input_id_dict["target_input_ids"])
        combined_labels = torch.cat((source_labels, target_labels), dim=1)
        # ignore eos token id
        if input_ids[0, -1] == self.tokenizer.eos_token_id:
            combined_labels[:, -1] = -100

        # Track gold alignment labels (used in metrics)
        if not reverse:
            self.sure.append(set())

        # Fill label matrix from alignment pairs
        for source_target_pair in alignment.strip().split():
            sure_alignment = "-" in source_target_pair
            if not sure_alignment:
                logger.warning(f"Unrecognised alignment format: {source_target_pair}")
                continue  # skip failed alignment

            wsrc, wtgt = parse_single_alignment(
                source_target_pair,
                one_indexed=self.one_indexed,
                reverse=reverse,
            )

            if not reverse:
                if sure_alignment:
                    self.sure[-1].add((wsrc, wtgt))

            if wsrc < len(target_labels):
                target_labels[wsrc, :] = torch.where(
                    target_bpe2word == wtgt, 1, target_labels[wsrc, :]
                )

        if self.debug_mode:
            self.view_sure_alignments(self.sure[-1])
            self.view_labels(source_labels=source_labels, target_labels=target_labels)

        # Prepare final data structure
        if self.do_inference:
            # Inference mode - batch structure
            bpe2word_map = torch.cat((source_bpe2word, target_bpe2word), dim=0)
            data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": combined_labels,
                    "bpe2word_map": bpe2word_map,  # Needed for decoding alignments
                }
            )
        else:
            # Training mode - individual examples
            for input_id, label in zip(input_ids.tolist(), combined_labels.tolist()):
                data.append(
                    {
                        "input_ids": input_id,
                        "attention_mask": [1] * len(input_id),
                        "labels": label,
                    }
                )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_config = ModelConfig(model_name_or_path="FacebookAI/roberta-base")
    train_config = TrainConfig(experiment_name="trainer-test", mixed_precision="no")
    train_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/train.src",
        target_lines_path="data/cleaned_data/train.tgt",
        alignments_path="data/cleaned_data/train.talp",
        limit=35000,
    )
    eval_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/dev.src",
        target_lines_path="data/cleaned_data/dev.tgt",
        alignments_path="data/cleaned_data/dev.talp",
        limit=35000,
        do_inference=True,
    )
    dataloader_config = DataLoaderConfig(collate_fn=collate_fn_span)
    tok = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    train_data = AlignmentDatasetSilver(tokenizer=tok, **train_dataset_config.__dict__)
    eval_data = AlignmentDatasetSilver(tokenizer=tok, **eval_dataset_config.__dict__)
