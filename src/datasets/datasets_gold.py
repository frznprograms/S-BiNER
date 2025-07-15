from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from transformers import XLMRobertaTokenizer

from src.datasets.base_dataset import BaseDataset
from src.utils.pipeline_step import PipelineStep


@dataclass
class AlignmentDatasetGold(BaseDataset, PipelineStep):
    def __post_init__(self):
        super().__post_init__()
        self.one_indexed = True
        logger.success(f"{self.__class__.__name__} initialized successfully")

    def _prepare_labels(
        self,
        data: list,
        source_bpe2word: torch.Tensor,
        target_bpe2word: torch.Tensor,
        input_ids: torch.Tensor,
        input_id_dict: dict[str, Any],
        alignment: str,
        actual_target_len: int,
        reverse: bool = False,
    ):
        source_labels = torch.ones_like(input_id_dict["source_input_ids"]) * -100
        target_labels = torch.zeros(
            len(input_id_dict["source_input_ids"]), actual_target_len
        )
        labels = torch.cat((source_labels, target_labels), dim=1)

        # Process alignment pairs
        for source_target_pair in alignment.strip().split():
            # source target pair looks like "1:2/1"
            src_tgt_label = source_target_pair.split("/")
            source_target_idxs = src_tgt_label[0].split(":")
            source_idx, target_idx = source_target_idxs[0], source_target_idxs[1]
            pair_label = int(src_tgt_label[1])
            if pair_label == 0:
                continue  # do not accidentally convert 0s to 1s

            if not reverse:
                wsrc, wtgt = (
                    (int(source_idx), int(target_idx))
                    if self.one_indexed
                    else (int(source_idx) - 1, int(target_idx) - 1)
                )
                # alignment_tuple = (wsrc, wtgt)
                # self.sure[-1].add(alignment_tuple)
            else:
                wtgt, wsrc = (
                    (int(source_idx), int(target_idx))
                    if self.one_indexed
                    else (int(source_idx) - 1, int(target_idx) - 1)
                )

            # check validity of alignment indices
            if wsrc < len(target_labels):
                target_labels[wsrc, :] = torch.where(
                    target_bpe2word == wtgt, 1, target_labels[wsrc, :]
                )

        # Prepare final data structure
        if self.do_inference:
            # Inference mode - batch structure
            bpe2word_map = torch.cat((source_bpe2word, target_bpe2word), dim=0)
            data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": labels,
                    "bpe2wordmap": bpe2word_map,  # Needed for decoding alignments
                }
            )
        else:
            # Training mode - individual examples
            for input_id, label in zip(input_ids.tolist(), labels.tolist()):
                data.append(
                    {
                        "input_ids": input_id,
                        "attention_mask": [1] * len(input_id),
                        "labels": label[:512],
                    }
                )


if __name__ == "__main__":
    a = AlignmentDatasetGold(
        tokenizer=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base"),
        source_lines_path="data/raw_data/english.txt",
        target_lines_path="data/raw_data/chinese.txt",
        alignments_path="data/raw_data/alignment.txt",
        save=False,
    )

    # Debugging output; inlcude in execute() function and uncomment as needed:
    # if i == 0:
    #     print(f"Source input ids shape: {source_input_ids.shape}")
    #     print("\n")
    #     print(f"Source input ids: {source_input_ids}")
    #     print("\n")
    #     print(f"Target input ids shape: {target_input_ids.shape}")
    #     print("\n")
    #     print(f"Target input ids: {target_input_ids}")
    #     print("\n")
    #     print(f"Source labels: {source_labels}")
    #     print("\n")
    #     print(f"Source labels shape: {source_labels.shape}")
    #     print("\n")
    #     print(f"Target labels: {target_labels}")
    #     print("\n")
    #     print(f"Target labels shape: {target_labels.shape}")
    #     print("\n")
    #     print(f"Labels: {labels}")
    #     break
