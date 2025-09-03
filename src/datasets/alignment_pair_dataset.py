import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import pandas as pd
import torch
from easydict import EasyDict
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
from src.configs.model_config import ModelConfig
from src.configs.train_config import TrainConfig
from src.utils.decorators import timed_execution
from src.utils.helpers import delist_the_list

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


@dataclass
class AlignmentPairDataset(Dataset):
    tokenizer: PreTrainedTokenizer
    source_lines_path: str
    target_lines_path: str
    alignments_path: str
    dataloader_config: DataLoaderConfig
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
        self._is_prepared: bool = False
        # prepare data immediately
        self.run()
        if self.debug_mode:
            self._verify_data_types()

        logger.success(f"{self.__class__.__name__} initialized successfully")

    @logger.catch(reraise=True)
    def __len__(self):
        return len(self.data)

    @logger.catch(message="Unable to retrieve item", reraise=True)
    def __getitem__(self, index: int) -> dict[str, Union[str, torch.Tensor]]:
        source_sentence: str = self.source_sentences[index]
        target_sentence: str = self.target_sentences[index]
        alignments: str = self.alignments[index]

        # for efficiency, only convert to EasyDict when we want to get something
        item: dict[str, torch.Tensor] = EasyDict(self.data[index])
        input_ids: torch.Tensor = item.input_ids  # type: ignore
        source_token_to_word_mapping: torch.Tensor = item.source_token_to_word_mapping  # type: ignore
        target_token_to_word_mapping: torch.Tensor = item.target_token_to_word_mapping  # type: ignore        attention_mask: torch.Tensor = item.attention_mask  # type: ignore
        label_matrix: torch.Tensor = item.label_matrix  # type: ignore
        attention_mask: torch.Tensor = item.attention_mask  # type: ignore

        return {
            "source_sentence": source_sentence,
            "target_sentence": target_sentence,
            "alignments": alignments,
            "input_ids": input_ids,
            "source_word_ids": source_token_to_word_mapping,
            "target_word_ids": target_token_to_word_mapping,
            "attention_mask": attention_mask,
            "labels": label_matrix,
        }

    @timed_execution
    @logger.catch(message="Failed to complete dataset preparetion", reraise=True)
    def run(self):
        if self._is_prepared:
            answer = input(
                "Dataset already prepared, are you sure you wish to overried the dataset? [y/n]"
            )
            if answer.strip() == "n":
                logger.warning("Abandoning dataset preparation")
                return
        n = len(self.source_sentences)
        batch_size = self.dataloader_config.batch_size
        logger.info(f"Preparing dataset of {n} samples with batch size {batch_size}...")

        # clear any existing encoded data self.data.clear()
        self.reverse_data.clear()

        pbar = tqdm(total=n)
        for i in range(0, n, batch_size):
            batch_forward_res = self._prepare_data(start_index=i)
            batch_reverse_res = self._prepare_data(start_index=i, reverse=True)
            assert len(batch_forward_res) == len(batch_reverse_res)
            # just in case batch gets cut off (i.e. len of data not perfectly divisible by batch_size)
            # Convert batched results to individual entries
            curr_batch_size = batch_forward_res["source_input_ids"].size(0)
            for j in range(curr_batch_size):
                # Create individual forward entry
                forward_entry = {
                    "input_ids": torch.cat(
                        [
                            batch_forward_res["source_input_ids"][j],
                            batch_forward_res["target_input_ids"][j],
                        ]
                    ),
                    "source_token_to_word_mapping": batch_forward_res[
                        "source_token_to_word_mapping"
                    ][j],
                    "target_token_to_word_mapping": batch_forward_res[
                        "target_token_to_word_mapping"
                    ][j],
                    "attention_mask": torch.cat(
                        [
                            batch_forward_res["source_attn_mask"][j],
                            batch_forward_res["target_attn_mask"][j],
                        ]
                    ),
                    "label_matrix": batch_forward_res["labels"][j],
                }
                self.data.append(forward_entry)

                # Create individual reverse entry
                reverse_entry = {
                    "input_ids": torch.cat(
                        [
                            batch_reverse_res["source_input_ids"][j],
                            batch_reverse_res["target_input_ids"][j],
                        ]
                    ),
                    "source_token_to_word_mapping": batch_reverse_res[
                        "source_token_to_word_mapping"
                    ][j],
                    "target_token_to_word_mapping": batch_reverse_res[
                        "target_token_to_word_mapping"
                    ][j],
                    "attention_mask": torch.cat(
                        [
                            batch_reverse_res["source_attn_mask"][j],
                            batch_reverse_res["target_attn_mask"][j],
                        ]
                    ),
                    "label_matrix": batch_reverse_res["labels"][j],
                }
                self.reverse_data.append(reverse_entry)

            pbar.update(batch_size)
            if self.debug_mode:
                logger.debug(
                    f"Processed batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}"
                )
        self._is_prepared = True
        logger.success(
            f"Dataset preparation complete. Processed {len(self.source_sentences) // batch_size} batches."
        )

    def _prepare_data(
        self, start_index: int, reverse: bool = False
    ) -> dict[str, torch.Tensor]:
        # note: these sentences have already been split into words
        batch_size = self.dataloader_config.batch_size
        end_index = min(start_index + batch_size, len(self.source_sentences))

        source_lines: list[str] = self.source_sentences[start_index:end_index]
        target_lines: list[str] = self.target_sentences[start_index:end_index]
        alignments: list[str] = self.alignments[start_index:end_index]

        if reverse:
            source_lines, target_lines = target_lines, source_lines
            reverse_alignments = self._prepare_alignments(alignments, reverse=True)
            prepped_alignments = reverse_alignments
        else:
            prepped_alignments = self._prepare_alignments(alignments, reverse=False)

        if self.tokenizer.sep_token_id is None:
            self.tokenizer.add_special_tokens({"sep_token": self.context_sep})

        # Tokenize source lines
        source_encoding: BatchEncoding = self.tokenizer(
            [line.split() for line in source_lines],
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_sentence_length,
            return_tensors="pt",
        )

        # Tokenize target lines
        target_encoding: BatchEncoding = self.tokenizer(
            [line.split() for line in target_lines],
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_sentence_length,
            return_tensors="pt",
        )

        source_input_ids, source_attn_mask = self._prepare_inputs(source_encoding)
        target_input_ids, target_attn_mask = self._prepare_inputs(target_encoding)
        batched_label_matrices = self._prepare_label_matrices(
            source_lines=source_lines,
            target_lines=target_lines,
            alignments_list=prepped_alignments,
        )

        # Get word mappings and pad them to match the tokenized sequence length
        source_token_to_word_mapping = self._prepare_word_mappings(
            source_encoding, len(source_lines)
        )
        target_token_to_word_mapping = self._prepare_word_mappings(
            target_encoding, len(target_lines)
        )

        if self.debug_mode:
            # all debugging functions were designed to take batches, even if batch_size=1
            self._view_tokens(
                source_input_ids=source_input_ids, target_input_ids=target_input_ids
            )
            print(f"SOURCE token-to-word mapping: {source_token_to_word_mapping}")
            print(f"TARGET token-to-word mapping: {target_token_to_word_mapping}")
            self._view_encoded_texts(
                source_input_ids=source_input_ids,
                target_input_ids=target_input_ids,
                source_attn_mask=source_attn_mask,
                target_attn_mask=target_attn_mask,
            )
            self._view_label_matrices(label_matrices=batched_label_matrices)

        return {
            "source_input_ids": source_input_ids,
            "target_input_ids": target_input_ids,
            "labels": batched_label_matrices,  # type: ignore
            "source_token_to_word_mapping": source_token_to_word_mapping,
            "target_token_to_word_mapping": target_token_to_word_mapping,
            "source_attn_mask": source_attn_mask,
            "target_attn_mask": target_attn_mask,
        }  # type: ignore

    @logger.catch(message="Unable to prepare alignments", reraise=True)
    def _prepare_alignments(
        self, alignments: list[str], reverse: bool = False
    ) -> list[list[tuple[int, int]]]:
        new_list = []
        for i in range(len(alignments)):
            alignments_list = alignments[i].split()
            temp = []
            for pair in alignments_list:
                orig_source, orig_target = pair.split("-")
                if reverse:
                    new_source, new_target = orig_target, orig_source
                    temp.append((int(new_source), int(new_target)))
                else:
                    temp.append((int(orig_source), int(orig_target)))
            new_list.append(temp)

        return new_list

    @logger.catch(message="Unale to prepare input ids", reraise=True)
    def _prepare_inputs(
        self, encoded: BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = encoded["input_ids"]
        attn_mask = encoded["attention_mask"]
        # we will not prepare token type ids

        #        sep_token_id = self.tokenizer.sep_token_id
        #
        #        # Find separator positions
        #        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        #        token_type_ids = torch.zeros_like(input_ids)
        #
        #        if len(sep_positions) > 0:
        #            first_sep_pos = sep_positions[0]
        #            # Everything after the first separator for roberta should be token_type_id = 1
        #            token_type_ids[first_sep_pos + 1 :] = 1
        #        # token_type_ids = token_type_ids.unsqueeze(0) no need to add batch size oops
        #

        return input_ids, attn_mask  # type: ignore

    @logger.catch(message="Unable to prepare label matrices", reraise=True)
    def _prepare_label_matrices(
        self,
        source_lines: list[str],
        target_lines: list[str],
        alignments_list: list[list[tuple[int, int]]],
    ) -> list[torch.Tensor]:
        label_matrices = []
        for source_line, target_line, alignments in zip(
            source_lines, target_lines, alignments_list
        ):
            # assume both source and target lines are already split into words
            # print(f"Source line for matrix formation: {source_line.split()}")
            # print(f"Target line for matrix formation: {target_line.split()}")
            # print(f"Length of source line (in words): {len(source_line.split())}")
            # print(f"Length of target line (in words): {len(target_line.split())}")
            source_dim = len(source_line.split())
            target_dim = len(target_line.split())
            label_matrix = torch.zeros((source_dim, target_dim), dtype=torch.float)
            for source_i, target_j in alignments:
                if source_i < source_dim and target_j < target_dim:
                    label_matrix[source_i, target_j] = 1.0
            label_matrices.append(label_matrix)

        return label_matrices

    @logger.catch(message="Unable to view tokens", reraise=True)
    def _view_tokens(
        self, source_input_ids: torch.Tensor, target_input_ids: torch.Tensor
    ) -> None:
        print("Now showing tokens")
        batch_size = source_input_ids.shape[0]

        # Iterate over each entry in the batch
        for i in range(batch_size):
            print("=" * 50)
            print(f"BATCH ENTRY {i + 1}")
            print("=" * 50)

            # Extract single entries from batch
            source_ids_single = source_input_ids[i]
            target_ids_single = target_input_ids[i]

            # Convert to tokens and decode
            source_tokens = self.tokenizer.convert_ids_to_tokens(
                source_ids_single  # type: ignore
            )
            target_tokens = self.tokenizer.convert_ids_to_tokens(
                target_ids_single  # type:ignore
            )
            source_text = self.tokenizer.decode(source_ids_single)
            target_text = self.tokenizer.decode(target_ids_single)

            print("Source side")
            print("-" * 25)
            print(f"Original source text: {source_text}")
            print(f"Ids to tokens: {source_tokens}")
            print("-" * 25)
            print("Target side")
            print("-" * 25)
            print(f"Original target text: {target_text}")
            print(f"Ids to tokens: {target_tokens}")

        print("=" * 50)

    @logger.catch(message="Unable to view encoded text", reraise=True)
    def _view_encoded_texts(
        self,
        source_input_ids: torch.Tensor,
        source_attn_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attn_mask: torch.Tensor,
    ) -> None:
        print("Now showing the encoded text output")
        print("=" * 50)
        print("Source encoding")
        print("-" * 25)
        print(f"Source input ids shape: {source_input_ids.shape}")
        print(f"Source input ids: {source_input_ids}")
        print(f"Source attention mask shape: {source_attn_mask.shape}")
        print(f"Source attention mask: {source_attn_mask}")
        print("=" * 50)
        print("Target encoding")
        print("-" * 25)
        print(f"Target input ids shape: {target_input_ids.shape}")
        print(f"Target input ids: {target_input_ids}")
        print(f"Target attention mask shape: {target_attn_mask.shape}")
        print(f"Target attention mask: {target_attn_mask}")
        print("=" * 50)

    @logger.catch(message="Unable to view label matrices", reraise=True)
    def _view_label_matrices(self, label_matrices: list[torch.Tensor]):
        print("Now showing batched label matrices")
        print(f"Batched matrices size: {len(label_matrices)}")
        print("=" * 50)
        for matrix in label_matrices:
            print(f"Matrix shape: {matrix.shape}")
            print(matrix)
            print("-" * 50)

    @logger.catch(message="Unable to verify datatypes", reraise=True)
    def _verify_data_types(self) -> None:
        for entry in self.data:
            for key, value in entry.items():
                print(f"{key}: {type(value)}")

    @logger.catch(message="Unable to prepare word mappings", reraise=True)
    def _prepare_word_mappings(
        self, encoding: BatchEncoding, batch_size: int
    ) -> torch.Tensor:
        """
        Prepare word mappings as tensors with proper padding.
        """
        # Get the sequence length from the input_ids (which are already padded)
        seq_length = encoding["input_ids"].size(1)  # type: ignore

        # Initialize tensor with -1 (padding value)
        word_mappings = torch.full((batch_size, seq_length), -1, dtype=torch.long)

        # Fill in the actual word mappings
        for i in range(batch_size):
            word_ids = encoding.word_ids(batch_index=i)
            for j, word_id in enumerate(word_ids):
                if j < seq_length:  # Safety check
                    word_mappings[i, j] = word_id if word_id is not None else -1

        return word_mappings

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


if __name__ == "__main__":
    model_config = ModelConfig(model_name_or_path="FacebookAI/roberta-base")
    train_config = TrainConfig(experiment_name="trainer-test", mixed_precision="no")
    train_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/train.src",
        target_lines_path="data/cleaned_data/train.tgt",
        alignments_path="data/cleaned_data/train.talp",
        limit=1000,
        debug_mode=False,
    )
    dataloader_config = DataLoaderConfig()  # just use default batch_size=4

    tok = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, add_prefix_space=True
    )
    d = AlignmentPairDataset(
        tokenizer=tok,
        **train_dataset_config.__dict__,
        dataloader_config=dataloader_config,
    )
    print("=" * 50)
    generated_data = d.data
    for elem in generated_data:
        for key, value in elem.items():
            print(f"{key}: {value.shape}")
    print("=" * 50)
