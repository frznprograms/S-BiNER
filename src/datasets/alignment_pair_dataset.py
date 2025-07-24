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
    context_sep: Optional[str] = " <SEP> "
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
        label_matrix: torch.FloatTensor = item.label_matrix  # type: ignore

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
        # note: these sentences have already been split into words
        source_sentence: str = self.source_sentences[index]
        target_sentence: str = self.target_sentences[index]
        alignments: str = self.alignments[index]

        # returns dict[str, torch.Tensor] directly
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

        source_len = len(source_sentence.split())
        target_len = len(target_sentence.split())
        label_matrix = self._prepare_label_matrix(
            dim1=source_len, dim2=target_len, sentence_alignments=alignments
        )

        if self.debug_mode:
            print(f"Encoding shape: {encoded.shape}")
            print("Encodings:")
            print(encoded)

        # prepare input ids and attention mask
        input_ids = encoded["input_ids"].squeeze()  # type: ignore
        attention_mask = encoded["attention_mask"].squeeze()  # type: ignore
        token_type_ids = encoded.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze()

        source_mask = (
            (token_type_ids == 0)
            if token_type_ids is not None
            else self._infer_mask(source_sentence, encoded)
        )
        target_mask = (
            (token_type_ids == 1)
            if token_type_ids is not None
            else self._infer_mask(target_sentence, encoded)
        )

        # TODO: bpe2word encoding so we can map tokens back to the origin word

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "src_mask": source_mask,
            "tgt_mask": target_mask,
            "labels": label_matrix,
        }

    @logger.catch(message="Unable to prepare labels", reraise=True)
    def _prepare_label_matrix(self, dim1: int, dim2: int, sentence_alignments: str):
        label_matrix = torch.zeros((dim1, dim2), dtype=torch.float)
        for source_i, target_j in sentence_alignments.split():
            if int(source_i) < dim1 and int(target_j) < dim2:
                label_matrix[int(source_i), int(target_j)] = 1.0

    @logger.catch(message="Unable to infer mask", reraise=True)
    def _infer_mask(self, sentence, encoded):
        # TODO: consider edge cases: what happens if tokenizer does not have a separator id?
        # TODO: what happens in xlm case where two separators are used e.g. </s></s>
        # fallback method: split using sep token if no token_type_ids
        sep_token_id = self.tokenizer.sep_token_id
        input_ids = encoded["input_ids"].squeeze().tolist()
        sep_indices = [i for i, t in enumerate(input_ids) if t == sep_token_id]
        mask = torch.zeros(len(input_ids), dtype=torch.bool)
        # get source tokens
        if len(sep_indices) >= 2:
            start = 1
            end = sep_indices[0]
            mask[start:end] = True

        return mask

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
