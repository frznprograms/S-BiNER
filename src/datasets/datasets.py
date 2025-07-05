import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from configs.pipeline_configs import PipelineConfig
from src.utils.logger_config import LoggedPipelineStep
from src.utils.pipeline_step import PipelineStep


@dataclass
class AlignmentDataset(PipelineStep, LoggedPipelineStep):
    tokenizer: PreTrainedTokenizer
    source_lines: list[str]
    target_lines: list[str]
    alignments: list[str]
    config: PipelineConfig
    one_indexed: bool = True
    context_sep: Optional[str] = " [WORD_SEP] "
    do_inference: bool = False

    data: list = field(default_factory=list, init=False)
    reverse_data: list = field(default_factory=list, init=False)
    sure: list = field(default_factory=list, init=False)

    def __post_init__(self):
        PipelineStep.__init__(self, self.config)
        if self.config is not None:
            LoggedPipelineStep.__init__(self, self.config)

        logger.info(f"Starting {self.step_name} step...")
        logger.success("AlignmentDataset initialised.")

        logger.info("Preparing dataset...")
        self.execute(
            source_lines=self.source_lines,
            target_lines=self.target_lines,
            alignments=self.alignments,
            data=self.data,
        )
        # self.execute(
        #     source_lines=self.source_lines,
        #     target_lines=self.target_lines,
        #     alignments=self.alignments,
        #     data=self.reverse_data,
        #     reverse=True,
        # )
        if not self.do_inference:
            self.data = self.data + self.reverse_data

    def execute(
        self,
        source_lines: list[str],
        target_lines: list[str],
        alignments: list[str],
        data: list[dict],
        reverse: bool = False,
    ) -> None:
        progress_bar = tqdm(total=len(source_lines))
        for i, (source_line, target_line, alignment) in enumerate(
            zip(source_lines, target_lines, alignments)
        ):
            progress_bar.update(1)
            if reverse:
                source_line, target_line = target_line, source_line

            word_by_word_examples = []
            source_sentence, target_sentence = (
                source_line.strip().split(),
                target_line.strip().split(),
            )
            if not source_line or not target_line:
                logger.warning(f"Either source or target line was empty at index {i}")
                continue

            # generate samples for alignment of each word in sentence
            for j, unit in enumerate(source_sentence):
                word_by_word_examples.append(
                    source_sentence[:j]
                    + [self.context_sep]
                    + [source_sentence[j]]
                    + [self.context_sep]
                    + source_sentence[j + 1 :]
                )

            source_tokens: list[list[list[str]]] = [
                [self.tokenizer.tokenize(word) for word in sentence]
                for sentence in word_by_word_examples
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

            # get byte-pair mappings
            source_bpe2word = (
                torch.ones_like(source_input_ids[0]) * -1
            )  # multiply by -1 to tell model to ignore word boundaries in source text
            target_bpe2word = []
            for k, word_list in enumerate(target_tokens):
                target_bpe2word += [i for _ in word_list]
            target_bpe2word = torch.tensor(
                target_bpe2word + [-1]
            )  # -1 added to tell model to ignore last word

            # encoder-decoder-like input
            input_ids = torch.cat((source_bpe2word, target_bpe2word), dim=-1)[:256]
            source_labels = (
                torch.ones_like(source_input_ids) * -100
            )  # tells pytorch to ignore source labels in loss calculation
            target_labels = torch.zeros_like(target_input_ids)

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

                if wsrc < len(target_labels):
                    target_labels[wsrc, :] = torch.where(
                        target_bpe2word == wtgt, 1, target_labels[wsrc, :]
                    )

                if not reverse:
                    alignment_tuple = (wsrc, wtgt)
                    print(f"Alignment tuple: {alignment_tuple}")
                    self.sure[-1].add(alignment_tuple)

            labels = torch.cat((source_labels, target_labels), dim=1)

            if i == 0:
                print(f"Sure labesl: {self.sure[-1]}")

                break
