import gc
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from src.utils.decorators import timed_execution
from src.utils.helpers import delist_the_list


@dataclass
class BaseDataset(ABC):
    tokenizer: PreTrainedTokenizer
    source_lines_path: str
    target_lines_path: str
    alignments_path: str
    limit: Optional[int] = None
    one_indexed: bool = False
    context_sep: Optional[str] = " [WORD_SEP] "
    do_inference: bool = False
    log_output_dir: str = "logs"
    save: bool = False
    debug_mode: bool = True
    sure: list = field(default_factory=list, init=False)
    data: list = field(default_factory=list, init=False)
    reverse_data: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.source_lines = self.read_data(
            path=self.source_lines_path, limit=self.limit
        )
        self.target_lines = self.read_data(
            path=self.target_lines_path, limit=self.limit
        )
        self.alignments = self.read_data(path=self.alignments_path, limit=self.limit)

        logger.debug("Preparing dataset...")
        self.run()

        # Create bidirectional data if not doing inference
        if not self.do_inference:
            logger.debug("Combining forward and reverse data now...")
            self.data = self.data + self.reverse_data

        if self.save:
            self.save_data(self.data, "data/data.pt")
            self.save_data(self.reverse_data, "data/reverse_data.pt")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.do_inference:
            return self.data[item], self.reverse_data[item]
        else:
            return self.data[item]

    @property
    def alignment_sure(self) -> list[set[tuple[int, int]]]:
        return self.sure

    def run(self):
        self.prepare_data(
            source_lines=self.source_lines,
            target_lines=self.target_lines,
            alignments=self.alignments,
            data=self.data,
        )

        self.prepare_data(
            source_lines=self.source_lines,
            target_lines=self.target_lines,
            alignments=self.alignments,
            data=self.reverse_data,
            reverse=True,
        )

    @timed_execution
    @logger.catch(message="Failed to prepare dataset", reraise=True)
    def prepare_data(
        self,
        source_lines: list[str],
        target_lines: list[str],
        alignments: list[str],
        data: list[dict],
        reverse: bool = False,
    ):
        progress_bar = tqdm(total=len(source_lines))

        for i, (source_line, target_line, alignment) in enumerate(
            zip(source_lines, target_lines, alignments)
        ):
            progress_bar.update(1)
            # symmetrisation
            if reverse:
                source_line, target_line = target_line, source_line

            # Generate word-by-word examples
            word_by_word_examples, source_sentence, target_sentence = (
                self._generate_wbw_examples(
                    source_line=source_line, target_line=target_line
                )
            )
            # Prepare input IDs
            input_id_dict = self._prepare_input_ids(
                wbw_examples=word_by_word_examples, target_sentence=target_sentence
            )
            # Create combined input IDs
            input_ids = torch.cat(
                (input_id_dict["source_input_ids"], input_id_dict["target_input_ids"]),
                dim=1,
            )

            # Prepare BPE to word mappings
            source_bpe2word = torch.ones_like(input_id_dict["source_input_ids"][0]) * -1
            target_bpe2word = self._create_target_bpe2word_mapping(input_id_dict)

            # Adjust target_bpe2word to match actual target length, just in case
            source_len = input_id_dict["source_input_ids"].shape[1]
            actual_target_len = input_ids.shape[1] - source_len
            target_bpe2word = target_bpe2word[:actual_target_len]

            if self.debug_mode:
                # BaseDataset.view_wbw_examples(examples=word_by_word_examples)
                # BaseDataset.view_input_id_dict(
                #     source_sentence=source_sentence,
                #     target_sentence=target_sentence,
                #     input_id_dict=input_id_dict,
                # )
                # BaseDataset.view_bpe2word_mappings(
                #     input_id_dict=input_id_dict,
                #     source_bpe2word=source_bpe2word,
                #     target_bpe2word=target_bpe2word,
                # )
                # return
                pass

            # Prepare labels (abstract method - implemented by child classes)
            self._prepare_labels(
                data=data,
                source_bpe2word=source_bpe2word,
                target_bpe2word=target_bpe2word,
                input_ids=input_ids,
                input_id_dict=input_id_dict,
                alignment=alignment,
                actual_target_len=actual_target_len,
                reverse=reverse,
            )

            # Cleanup
            del input_ids, target_bpe2word
            if i % 20 == 0:
                gc.collect()

    @logger.catch(
        message="Error in generating target byte-pair encodings", reraise=True
    )
    def _create_target_bpe2word_mapping(
        self, input_id_dict: dict[str, Any]
    ) -> torch.Tensor:
        target_bpe2word = []
        for k, tokens_for_word in enumerate(input_id_dict["target_tokens"]):
            target_bpe2word += [k for _ in tokens_for_word]

        return torch.Tensor(
            target_bpe2word + [-1]
        )  # -1 to remove last word from consideration

    @logger.catch(
        message="Error in generating word-by-word source examples", reraise=True
    )
    def _generate_wbw_examples(
        self, source_line: str, target_line: str
    ) -> tuple[list[list[str]], list[str], list[str]]:
        res = []
        source_sentence: list[str]
        target_sentence: list[str]
        source_sentence, target_sentence = (
            source_line.strip().split(),
            target_line.strip().split(),
        )

        if not source_line or not target_line:
            logger.warning("Either source line or target line is missing.")

        for j, unit in enumerate(source_sentence):
            res.append(
                source_sentence[:j]
                + [self.context_sep]
                + [source_sentence[j]]
                + [self.context_sep]
                + source_sentence[j + 1 :]
            )

        return res, source_sentence, target_sentence

    @logger.catch(message="Error in preparing input ids", reraise=True)
    def _prepare_input_ids(
        self, wbw_examples: list[list[str]], target_sentence: list[str]
    ) -> dict[str, Any]:
        source_tokens: list[list[list[str]]] = [
            [self.tokenizer.tokenize(word) for word in sentence]
            for sentence in wbw_examples
        ]
        target_tokens: list[list[str]] = [
            self.tokenizer.tokenize(word) for word in target_sentence
        ]

        source_w2id: list[list[list[int]]] = [
            [self.tokenizer.convert_tokens_to_ids(subunit) for subunit in token]
            for token in source_tokens
        ]  # type: ignore

        target_w2id: list[list[int]] = [
            self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens
        ]  # type: ignore

        source_input_ids = torch.tensor(
            [
                self.tokenizer.prepare_for_model(
                    list(itertools.chain(*word_ids)),
                    truncation=True,
                    max_length=512,
                )["input_ids"]
                for word_ids in source_w2id
            ]
        )

        target_input_ids = self.tokenizer.prepare_for_model(
            list(itertools.chain(*target_w2id)),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )["input_ids"][1:]  # type: ignore | remove first cls token
        target_input_ids = target_input_ids.repeat(len(source_input_ids), 1)

        return {
            "source_tokens": source_tokens,
            "source_w2id": source_w2id,
            "source_input_ids": source_input_ids,
            "target_tokens": target_tokens,
            "target_w2id": target_w2id,
            "target_input_ids": target_input_ids,
        }

    def read_data(self, path: str, limit: Optional[int]) -> list[str]:
        data = delist_the_list(pd.read_csv(path, sep="\t").values.tolist())
        if limit is None:
            limit = len(data)
        data = data[:limit]

        return data

    def save_data(self, data: Any, save_path: str, format: str = "pt") -> None:
        if format == "csv":
            data.to_csv(save_path)
        elif format == "pt":
            torch.save(data, save_path)
        logger.success("Data saved successfully.")

    @abstractmethod
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
        pass

    # helper static functions for debugging
    @staticmethod
    def view_input_id_dict(
        source_sentence: list[str],
        target_sentence: list[str],
        input_id_dict: dict[str, Any],
    ) -> None:
        logger.debug("Showing input_id_dict")
        print("=" * 50)
        print(f"Source sentence: {source_sentence}")
        print(f"Source tokens: {input_id_dict['source_tokens']}")
        print(f"Source input ids shape: {input_id_dict['source_input_ids'].shape}")
        print("=" * 50)
        print("Source input ids:")
        print("[")
        for elem in input_id_dict["source_input_ids"]:
            print(f"\t{elem}")
        print("]")
        print("=" * 50)
        print("Source w2ids:")
        print("[")
        for elem in input_id_dict["source_w2id"]:
            print(f"\t{elem}")
        print("]")

        print("=" * 50)

        print(f"Target sentence: {target_sentence}")
        print(f"Target tokens: {input_id_dict['target_tokens']}")
        print(f"Target input ids shape: {input_id_dict['target_input_ids'].shape}")
        print("=" * 50)
        print("Target input ids:")
        print("[")
        for elem in input_id_dict["target_input_ids"]:
            print(f"\t{elem}")
        print("]")
        print("=" * 50)
        print("Target w2ids:")
        print("[")
        for elem in input_id_dict["target_w2id"]:
            print(f"\t{elem}")
        print("]")
        print("=" * 50)

    @staticmethod
    def view_wbw_examples(examples: list[list[str]]) -> None:
        logger.debug("Showing word_by_word_examples")
        print("=" * 50)
        print("Word by word examples:")
        for example in examples:
            print(f"\t{example}")
        print("=" * 50)

    @staticmethod
    def view_bpe2word_mappings(
        input_id_dict: dict[str, Any],
        source_bpe2word: torch.Tensor,
        target_bpe2word: torch.Tensor,
    ) -> None:
        logger.debug("Showing bpe2word mappings")
        source_input_ids = input_id_dict["source_input_ids"]
        target_input_ids = input_id_dict["target_input_ids"]
        print("=" * 50)
        print(f"Source input ids shape: {source_input_ids.shape}")
        print("=" * 50)
        print("Source input ids:")
        print("[")
        for elem in source_input_ids:
            print(f"\t{elem}")
        print("]")
        print("=" * 50)
        print("Source byte-pair encodings:")
        print(source_bpe2word)
        print("=" * 50)
        print(f"Target input ids shape: {target_input_ids.shape}")
        print("=" * 50)
        print("Target input ids:")
        print("[")
        for elem in target_input_ids:
            print(f"\t{elem}")
        print("]")
        print("=" * 50)
        print("Target byte-pair encodings:")
        print(target_bpe2word)
        print("=" * 50)
