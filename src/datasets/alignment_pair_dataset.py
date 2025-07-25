import os
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd
import torch
from easydict import EasyDict
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

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
    context_sep: str = " <SEP> "
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
        input_ids: torch.LongTensor = item.input_ids  # type: ignore
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
    def run(self, track_memory_usage: bool = True):
        pbar = tqdm(total=len(self.source_sentences))
        if track_memory_usage:
            tracemalloc.start()

        for i in range(len(self.source_sentences)):
            try:
                entry = self.prepare_data(index=i, reverse=False)
                reversed_entry = self.prepare_data(index=i, reverse=True)
                self.data.append(entry)
                self.reverse_data.append(reversed_entry)
            except Exception:
                pbar.write(f"Unable to complete data preparation at index {i}")
            pbar.update(1)

        # override to prevent excessive memory usage
        if self.do_inference:
            self.data = self.data + self.reverse_data

        if track_memory_usage:
            current, peak = tracemalloc.get_traced_memory()
            pbar.write(
                f"Current usage: {current / 1e6:.2f} MB | Peak usage: {peak / 1e6:.2f} MB"
            )

        logger.success(
            f"Prepared dataset for tasking inference set to {self.do_inference}"
        )

    @logger.catch(message="Unable to prepare dataset", reraise=True)
    def prepare_data(self, index: int, reverse: bool = False):
        # note: these sentences have already been split into words
        source_sentence: str = self.source_sentences[index]
        target_sentence: str = self.target_sentences[index]
        alignments: str = self.alignments[index]

        if reverse:
            source_sentence, target_sentence = target_sentence, source_sentence
            # need to reverse the alignments as well
            reverse_alignments = self._prepare_alignments(alignments, reverse=True)
            prepped_alignments = reverse_alignments
        else:
            prepped_alignments = self._prepare_alignments(alignments, reverse=False)

        if self.tokenizer.sep_token_id is None:
            self.tokenizer.add_special_tokens({"sep_token": self.context_sep})

        encoded = self.tokenizer(
            source_sentence,
            target_sentence,
            is_split_into_words=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_sentence_length,
            return_token_type_ids=True,
        )

        # prepare input ids and attention mask
        input_ids, attention_mask, token_type_ids = self._prepare_inputs(encoded)

        source_len = len(source_sentence.split())
        target_len = len(target_sentence.split())
        label_matrix = self._prepare_label_matrix(
            dim1=source_len, dim2=target_len, sentence_alignments=prepped_alignments
        )

        source_mask = (
            (token_type_ids == 0)
            if token_type_ids is not None
            else self._infer_mask(encoded)
        )
        target_mask = (
            (token_type_ids == 1)
            if token_type_ids is not None
            else self._infer_mask(encoded)
        )

        # reverse mapping so we can map tokens back to the origin word
        combined_token_to_word_mapping = encoded.word_ids()

        if self.debug_mode:
            self._view_tokens(input_ids)
            print(f"Token-to_word mapping: {combined_token_to_word_mapping}")
            self._view_encoded_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "src_mask": source_mask,
            "tgt_mask": target_mask,
            "labels": label_matrix,
            "word_to_token_mapping": combined_token_to_word_mapping,
        }

    @logger.catch(message="Unable to prepare reverse alignments", reraise=True)
    def _prepare_alignments(
        self, alignments: str, reverse: bool = False
    ) -> list[tuple[int, int]]:
        alignments_list = alignments.split()
        new_list = []
        for pair in alignments_list:
            orig_source, orig_target = pair.split("-")
            if reverse:
                new_source, new_target = orig_target, orig_source
                new_list.append((int(new_source), int(new_target)))
            else:
                new_list.append((int(orig_source), int(orig_target)))

        return new_list

    @logger.catch(message="Unable to prepare labels", reraise=True)
    def _prepare_label_matrix(
        self, dim1: int, dim2: int, sentence_alignments: list[tuple[int, int]]
    ):
        label_matrix = torch.zeros((dim1, dim2), dtype=torch.float)
        for source_i, target_j in sentence_alignments:
            if source_i < dim1 and target_j < dim2:
                label_matrix[source_i, target_j] = 1.0

    @logger.catch(message="Unable to infer mask", reraise=True)
    def _infer_mask(self, encoded: BatchEncoding):
        # fallback method: split using sep token if no token_type_ids
        sep_token_id = self.tokenizer.sep_token_id
        input_ids = encoded["input_ids"].squeeze().tolist()  # type: ignore
        sep_indices = [i for i, t in enumerate(input_ids) if t == sep_token_id]
        mask = torch.zeros(len(input_ids), dtype=torch.bool)
        # get source tokens for cases like </s></s>
        if len(sep_indices) >= 2:
            start = 1
            end = sep_indices[1]
            mask[start:end] = True

        return mask

    @logger.catch(message="Unable to prepare inputs", reraise=True)
    def _prepare_inputs(self, encoded: BatchEncoding):
        input_ids = encoded["input_ids"].squeeze()  # type: ignore
        attention_mask = encoded["attention_mask"].squeeze()  # type: ignore
        sep_token_id = self.tokenizer.sep_token_id

        # Find separator positions
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        token_type_ids = torch.zeros_like(input_ids)

        if len(sep_positions) > 0:
            first_sep_pos = sep_positions[0]
            # Everything after the first separator for roberta should be token_type_id = 1
            token_type_ids[first_sep_pos + 1 :] = 1
        # token_type_ids = token_type_ids.unsqueeze(0) no need to add batch size oops

        return input_ids, attention_mask, token_type_ids

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

    @logger.catch(message="Unable to view sentence and its tokens", reraise=True)
    def _view_tokens(self, input_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        combined_text = self.tokenizer.decode(input_ids)
        print("=" * 50)
        print(f"Original sentence: {combined_text}")
        print(f"Tokens: {tokens}")
        print("=" * 50)

    @logger.catch(message="Unable to view encoded text", reraise=True)
    def _view_encoded_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        print("Now showing the encoded text output")
        print("=" * 50)
        print(f"Encoded input ids shape: {input_ids.shape}")
        print(f"Encoded input ids: {input_ids}")
        print("=" * 50)
        print(f"Encoded attention mask shape: {attention_mask.shape}")
        print(f"Encoded attention mask: {attention_mask}")
        print("=" * 50)
        print(f"Encoded token type ids shape: {token_type_ids.shape}")
        print(f"Encoded token type ids: {token_type_ids}")
        print("=" * 50)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from src.configs.dataset_config import DatasetConfig
    from src.configs.model_config import ModelConfig
    from src.configs.train_config import TrainConfig

    model_config = ModelConfig(model_name_or_path="FacebookAI/roberta-base")
    train_config = TrainConfig(experiment_name="trainer-test", mixed_precision="no")
    train_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/train.src",
        target_lines_path="data/cleaned_data/train.tgt",
        alignments_path="data/cleaned_data/train.talp",
        limit=100000,
    )

    tok = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, add_prefix_space=True
    )
    d = AlignmentPairDataset(tokenizer=tok, **train_dataset_config.__dict__)
    d.run()
