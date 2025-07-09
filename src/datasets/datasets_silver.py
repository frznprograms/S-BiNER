import itertools
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from configs.pipeline_configs import PipelineConfig
from src.utils.helpers import load_data
from configs.logger_config import LoggedPipelineStep
from src.utils.pipeline_step import PipelineStep


@dataclass
class AlignmentDatasetSilver(PipelineStep, LoggedPipelineStep):
    tokenizer: PreTrainedTokenizer
    source_lines: list[str]
    target_lines: list[str]
    config: PipelineConfig
    one_indexed: bool = False
    context_sep: Optional[str] = " <SEP> "
    do_inference: bool = False
    save_data: bool = False

    data: list = field(default_factory=list, init=False)
    reverse_data: list = field(default_factory=list, init=False)
    sure: list = field(default_factory=list, init=False)
    alignments: list[str] = field(
        default_factory=list, init=False
    )  # TODO: figure out awesome align output

    def __post_init__(self):
        PipelineStep.__init__(self, self.config)
        LoggedPipelineStep.__init__(self, self.config)

        logger.info(f"Starting {self.step_name} step...")
        logger.success("AlignmentDatasetUnsupervised initialised.")

        self.execute(
            source_lines=self.source_lines,
            target_lines=self.target_lines,
            alignments=self.alignments,
        )

        self.execute(
            source_lines=self.source_lines,
            target_lines=self.target_lines,
            alignments=self.alignments,
            reverse=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.do_inference:
            return self.data[item], self.reverse_data[item]
        else:
            return self.data[item]

    def execute(
        self,
        source_lines: list[str],
        target_lines: list[str],
        alignments: list[str],
        reverse: bool = False,
    ):
        progress_bar = tqdm(total=len(source_lines))
        for i, (source_line, target_line, alignment) in enumerate(
            zip(source_lines, target_lines, alignments)
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

            source_bpe2word = torch.ones_like(input_id_dict["source_input_ids"]) * -1

            target_bpe2word = []
            for k, word_list in enumerate(input_id_dict["target_tokens"]):
                target_bpe2word += [k for _ in word_list]

            target_bpe2word = torch.Tensor(
                target_bpe2word + [-1]
            )  # -1 added to remove last word from model's consideration

            # adjust target_bpe2word to match the actual target length in input_ids
            source_len = input_id_dict["source_input_ids"].shape[1]
            actual_target_len = input_ids.shape[1] - source_len
            target_bpe2word = target_bpe2word[:actual_target_len]

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

    def _prepare_labels(
        self, input_id_dict: dict[str, Any], alignment: str, actual_target_len: int
    ):
        source_labels = torch.ones_like(input_id_dict["source_input_ids"]) * -100
        target_labels = torch.zeros(
            len(input_id_dict["source_input_ids"]), actual_target_len
        )

        for source_target_pair in alignment.strip().split():
            src_tgt_label = source_target_pair.split("-")
            source_idx = src_tgt_label[0]
            target_idx = src_tgt_label[1]
            # TODO: continue from here
