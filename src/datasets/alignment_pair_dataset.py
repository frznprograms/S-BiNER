import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd
import torch
from easydict import EasyDict
from loguru import logger
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from src.utils.decorators import timed_execution
from src.utils.helpers import delist_the_list

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


@dataclass
class AlignmentPairDataset(Dataset):
    tokenizer: PreTrainedTokenizer
    source_lines_path: str
    target_lines_path: str
    alignments_path: str
    limit: Optional[int] = None
    one_indexed: bool = False
    context_sep: Optional[str] = " <S> "
    do_inference: bool = False
    log_output_dir: str = "logs"
    max_sentence_length: int = 512
    save: bool = False
    debug_mode: bool = False
    save_dir: str = "output"
    sure: list = field(default_factory=list, init=False)
    data: list[dict[str, torch.Tensor]] = field(default_factory=list, init=False)
    reverse_data: list[dict[str, torch.Tensor]] = field(
        default_factory=list, init=False
    )

    # TODO: add context sep to tokenizer
    # TODO: find out if tensor types are correct -> does it all become FloatTensor anyway?

    def __post_init__(self):
        self.source_sentences: list[str] = self.read_data(
            path=self.source_lines_path, limit=self.limit
        )
        self.target_sentences: list[str] = self.read_data(
            path=self.target_lines_path, limit=self.limit
        )
        self.alignments: list[str] = self.read_data(
            path=self.alignments_path, limit=self.limit
        )
        logger.success(f"{self.__class__.__name__} initialized successfully")

    @logger.catch(reraise=True)
    def __len__(self):
        return len(self.data)

    @logger.catch(message="Unable to retrieve item", reraise=True)
    def __getitem__(self, index: int) -> dict[str, Union[str, torch.Tensor]]:
        source_sentence: str = self.source_sentences[index]
        target_sentence: str = self.target_sentences[index]
        alignments: str = self.alignments[index]

        item: dict[str, torch.Tensor] = EasyDict(self.data[index])
        input_ids: torch.FloatTensor = item.input_ids  # type: ignore
        source_mask: torch.BoolTensor = item.source_mask  # type: ignore
        target_mask: torch.BoolTensor = item.target_mask  # type: ignore
        attention_mask: torch.IntTensor = item.attention_mask  # type: ignore
        label_matrix: torch.IntTensor = item.label_matrix  # type: ignore

        return {
            "source_sentence": source_sentence,
            "target_sentence": target_sentence,
            "alignments": alignments,
            "input_ids": input_ids,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "attention_mask": attention_mask,
            "labels": label_matrix,
        }

    @timed_execution
    @logger.catch(message="Unable to prepare dataset", reraise=True)
    def prepare_data(self, index: int):
        source_sentence: str = self.source_sentences[index]
        target_sentence: str = self.target_sentences[index]
        alignments: str = self.alignments[index]

        # TODO: find out if this returns the word ids immediately
        encoded = self.tokenizer(
            source_sentence,
            target_sentence,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_sentence_length,
            return_token_type_ids=True,
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        token_type_ids = encoded.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze()

    @logger.catch(message="Unable to read data", reraise=True)
    def read_data(self, path: str, limit: Optional[int]) -> list[str]:
        data = delist_the_list(pd.read_csv(path, sep="\t").values.tolist())
        if limit is not None:
            data = data[:limit]

        return data

    @logger.catch(message="Unable to save data", reraise=True)
    def save_data(self, data: Any, save_path: str, format: str = "pt") -> None:
        if format == "csv":
            data.to_csv(save_path)
        elif format == "pt":
            torch.save(data, save_path)
        logger.success("Data saved successfully.")
