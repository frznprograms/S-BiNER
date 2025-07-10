import gc
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import XLMRobertaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from configs.logger_config import LoggedProcess
from src.datasets.base_dataset import BaseDataset
from src.utils.decorators import timed_execution
from src.utils.helpers import delist_the_list


@dataclass
class AlignmentDatasetGold(BaseDataset, LoggedProcess):
    tokenizer: PreTrainedTokenizer
    source_lines_path: str = "data/raw_data/english.txt"
    target_lines_path: str = "data/raw_data/chinese.txt"
    alignments_path: str = "data/raw_data/alignments.txt"
    one_indexed: bool = True
    context_sep: Optional[str] = " [WORD_SEP] "
    do_inference: bool = False
    log_output_dir: str = "/logs"
    save: bool = True

    data: list = field(default_factory=list, init=False)
    reverse_data: list = field(default_factory=list, init=False)
    sure: list = field(default_factory=list, init=False)

    def __post_init__(self):
        LoggedProcess.__init__(self, output_dir=self.log_output_dir)
        self.source_lines = self.read_data(path=self.source_lines_path)
        self.target_lines = self.read_data(path=self.target_lines_path)
        self.alignments = self.read_data(path=self.alignments_path)
        logger.success("AlignmentDatasetGold initialised.")

        logger.info("Preparing dataset...")
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

        # create bidirectional data
        if not self.do_inference:
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

    @timed_execution
    def prepare_data(
        self,
        source_lines: list[str],
        target_lines: list[str],
        alignments: list[str],
        data: list[dict],
        reverse: bool = False,
    ) -> None:
        progress_bar = tqdm(total=len(source_lines))
        for i, (source_line, target_line, alignment) in enumerate(
            zip(source_lines, target_lines, alignments),
        ):
            progress_bar.update(1)

            if reverse:
                source_line, target_line = target_line, source_line

            word_by_word_examples, source_sentence, target_sentence = (
                self._generate_wbw_examples(
                    source_line=source_line, target_line=target_line
                )
            )

            input_id_dict = self._prepare_input_ids(
                wbw_examples=word_by_word_examples, target_sentence=target_sentence
            )

            # encoder-decoder-like input
            input_ids = torch.cat(
                (input_id_dict["source_input_ids"], input_id_dict["target_input_ids"]),
                dim=1,
            )

            # get byte-pair mappings
            source_bpe2word = (
                torch.ones_like(input_id_dict["source_input_ids"]) * -1
            )  # multiply by -1 to tell model to ignore word boundaries in source text

            target_bpe2word = []
            for k, word_list in enumerate(input_id_dict["target_tokens"]):
                target_bpe2word += [k for _ in word_list]

            target_bpe2word = torch.tensor(
                target_bpe2word + [-1]
            )  # -1 added to tell model to ignore last word

            # Adjust target_bpe2word to match the actual target length in input_ids:
            source_len = input_id_dict["source_input_ids"].shape[1]
            actual_target_len = input_ids.shape[1] - source_len
            target_bpe2word = target_bpe2word[:actual_target_len]

            source_labels = (
                torch.ones_like(input_id_dict["source_input_ids"]) * -100
            )  # tells pytorch to ignore source labels in loss calculation
            target_labels = torch.zeros(
                len(input_id_dict["source_input_ids"]), actual_target_len
            )

            if not reverse:
                self.sure.append((set()))

            # handle labels -> note that there are no possible alignments, only sure ones
            for source_target_pair in alignment.strip().split():
                # source target pair looks like "1:2/1"
                src_tgt_label = source_target_pair.split("/")
                source_target_idxs = src_tgt_label[0].split(":")
                source_idx, target_idx = source_target_idxs[0], source_target_idxs[1]
                pair_label = int(src_tgt_label[1])
                if pair_label == 0:
                    continue  # do not accidentally convert 0s to 1s

                if reverse:
                    wtgt, wsrc = (
                        (int(source_idx), int(target_idx))
                        if self.one_indexed
                        else (int(source_idx) - 1, int(target_idx) - 1)
                    )
                else:
                    wsrc, wtgt = (
                        (int(source_idx), int(target_idx))
                        if self.one_indexed
                        else (int(source_idx) - 1, int(target_idx) - 1)
                    )

                # # BOUNDS CHECKING
                # if wsrc < 0 or wsrc >= len(source_sentence):
                #     continue

                # if wtgt < 0 or wtgt >= len(target_sentence):
                #     continue

                # # Additional check for tensor bounds
                # if wsrc >= len(target_labels):
                #     continue

                # # Check if wtgt exists in target_bpe2word before using it
                # if wtgt not in target_bpe2word:
                #     continue

                # if wsrc < len(target_labels):
                #     target_labels[wsrc, :] = torch.where(
                #         target_bpe2word == wtgt, 1, target_labels[wsrc, :]
                #     )

                if not reverse:
                    alignment_tuple = (wsrc, wtgt)
                    self.sure[-1].add(alignment_tuple)

            labels = torch.cat((source_labels, target_labels), dim=1)

            # check that the last token of every sequence is ignored if it is the tokenizer eos token
            if input_ids[0, -1] == self.tokenizer.eos_token_id:
                labels[:, -1] = -100

            if self.do_inference:
                # Inference mode - batch structure:
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
                # Training mode - individual examples:
                for input_id, label in zip(input_ids.tolist(), labels.tolist()):
                    data.append(
                        {
                            "input_ids": input_id,
                            "attention_mask": [1] * len(input_id),
                            "labels": label[:256],
                        }
                    )

            del (
                input_ids,
                labels,
                input_id_dict["source_input_ids"],
                input_id_dict["target_input_ids"],
            )
            del source_labels, target_labels, target_bpe2word
            if i % 20 == 0:  # More frequent cleanup
                gc.collect()

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
                    max_length=256,
                )["input_ids"]
                for word_ids in source_w2id  # i.e. for each example
            ]
        )

        target_input_ids = self.tokenizer.prepare_for_model(
            list(itertools.chain(*target_w2id)),
            return_tensors="pt",
            truncation=True,
            max_length=256,
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

    def read_data(self, path: str, data_size: int = 21999) -> list[str]:
        data = delist_the_list(pd.read_csv(path, sep="\t").values.tolist())[:data_size]
        return data


if __name__ == "__main__":
    data_path_dict = {
        "src_data": "data/english.txt",
        "tgt_data": "data/chinese.txt",
        "align_data": "data/alignment.txt",
    }
    a = AlignmentDatasetGold(
        tokenizer=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base"),
        source_lines_path="data/raw_data/english.txt",
        target_lines_path="data/raw_data/chinese.txt",
        alignments_path="data/raw_data/alignment.txt",
        save=True,
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
