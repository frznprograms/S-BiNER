from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Union, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.binary_align_models import (
    RobertaModelForBinaryTokenClassification,
    XLMRobertaModelForBinaryTokenClassification,
)
from src.utils.pipeline_step import PipelineStep

AGG_FN = {
    "mean": torch.mean,
    "min": torch.min,
    "max": torch.max,
    "last": lambda x: x[-1],
    "first": lambda x: x[0],
}


@dataclass
class BinaryAlignEvaluator(PipelineStep):
    def run(
        self,
        dataloader: DataLoader,
        model: Union[
            XLMRobertaModelForBinaryTokenClassification,
            RobertaModelForBinaryTokenClassification,
        ],
        threshold: float,
        sure: list[set[tuple[int, int]]],
        device: str,
        mini_batch_size: int,
        bidirectional_combine_type: str,
        tk2word_prob: str,
        possible: Optional[list[set[tuple[int, int]]]] = None,
    ):
        return self._bidirectional_eval_span(
            dataloader=dataloader,
            model=model,
            threshold=threshold,
            sure=sure,
            possible=possible,
            device=device,
            mini_batch_size=mini_batch_size,
            bidirectional_combine_type=bidirectional_combine_type,
            tk2word_prob=tk2word_prob,
        )

    @logger.catch(message="Unable to get bidirectional metrics", reraise=True)
    def _bidirectional_eval_span(
        self,
        dataloader: DataLoader,
        model,
        threshold: float,
        sure: list[set[tuple[int, int]]],
        device: str,
        mini_batch_size: int,
        possible: Optional[list[set[tuple[int, int]]]] = None,
        bidirectional_combine_type: str = "intersection",
        tk2word_prob: str = "mean",
    ) -> tuple[float, float, float, float]:
        model.eval()
        logger.info("Running evaluations...")

        predicted_sure = []
        predicted_possible = []
        possible_threshold = max(0.1, threshold)

        for sample in tqdm(dataloader):
            if isinstance(sample, list):  # Bidirectional eval
                sample_1, sample_2 = sample
                sample_preds_1 = self._get_sample_probs(
                    sample_1, model, device, mini_batch_size, tk2word_prob
                )
                sample_preds_2 = self._get_sample_probs(
                    sample_2, model, device, mini_batch_size, tk2word_prob, reverse=True
                )
                all_probs = self._combine_pred_probs(
                    sample_preds_1, sample_preds_2, bidirectional_combine_type
                )
            else:  # Unidirectional
                all_probs = self._get_sample_probs(
                    sample, model, device, mini_batch_size, tk2word_prob
                )

            sure_set = {k for k, v in all_probs.items() if v >= threshold}
            predicted_sure.append(sure_set)

            if possible is not None:
                possible_set = {
                    k
                    for k, v in all_probs.items()
                    if possible_threshold <= v < threshold
                }
                predicted_possible.append(possible_set)

        metrics = self.calculate_metrics(
            sure,
            predicted_sure,
            possible,
            predicted_possible,
        )
        logger.info(
            f"AER: {metrics[2]:.4f}, Precision: {metrics[0]:.4f}, Recall: {metrics[1]:.4f}, F1: {metrics[3]:.4f}"
        )
        model.train()
        return metrics

    @logger.catch(
        message="Unable to get probabilities for a sample in the data", reraise=True
    )
    def _get_sample_probs(
        self,
        sample,
        model,
        device: str,
        mini_batch_size: int,
        tk2word_prob: str = "max",
        reverse: bool = False,
        softmax: Callable = nn.Softmax(dim=-1),
        sigmoid: Callable = nn.Sigmoid(),
    ) -> dict[tuple[int, int], float]:
        sample_preds = defaultdict(list)
        all_input_ids = sample["input_ids"].split(mini_batch_size)
        all_attention_mask = sample["attention_mask"].split(mini_batch_size)

        # Handle missing bpe2word_map gracefully
        if "bpe2word_map" not in sample:
            logger.warning("bpe2word_map not found in sample, creating dummy mapping")
            # Create a simple dummy mapping based on sequence length
            seq_len = sample["input_ids"].shape[1]
            bpe2word_map = (
                torch.arange(seq_len)
                .unsqueeze(0)
                .repeat(sample["input_ids"].shape[0], 1)
            )
        else:
            bpe2word_map = sample["bpe2word_map"]

        bpe2word_map = bpe2word_map.split(mini_batch_size)
        count = 0

        for input_ids, attention_mask, bpe2word_batch in zip(
            all_input_ids, all_attention_mask, bpe2word_map
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():  # Add no_grad for evaluation
                logits = model(input_ids, attention_mask=attention_mask).logits

            if logits.shape[2] == 2:
                probs = softmax(logits)[:, :, 1]
            else:
                probs = sigmoid(logits)[:, :, 0]

            for batch_idx in range(probs.size(0)):
                sentence_probs = probs[batch_idx]
                bpe2word = bpe2word_batch[batch_idx].tolist()

                # Handle potential length mismatches
                min_len = min(len(sentence_probs), len(bpe2word))
                for word_number in range(min_len):
                    word_prob = sentence_probs[word_number]
                    tgt = bpe2word[word_number]
                    if tgt != -1:
                        key = (tgt, count) if reverse else (count, tgt)
                        sample_preds[key].append(word_prob.item())
                count += 1

        return {
            k: AGG_FN[tk2word_prob](torch.tensor(v))
            for k, v in sample_preds.items()
            if v
        }

    @logger.catch(
        message="Unable to combine prediction probabilities for a sample", reraise=True
    )
    def _combine_pred_probs(
        self,
        preds_1: dict[tuple[int, int], float],
        preds_2: dict[tuple[int, int], float],
        method: str,
    ) -> dict[tuple[int, int], float]:
        if method in {"avg", "bidi_avg"}:
            return {
                k: (preds_1.get(k, 0) + preds_2.get(k, 0)) / 2
                for k in set(preds_1) | set(preds_2)
            }
        elif method == "union":
            return {**preds_1, **preds_2}
        elif method == "intersection":
            return {
                k: (preds_1[k] + preds_2[k]) / 2
                for k in preds_1.keys() & preds_2.keys()
            }
        else:
            logger.error(f"{method} not supported!")
            return {}

    @logger.catch(message="Unable to calculate metrics", reraise=True)
    def calculate_metrics(
        self,
        gold_sure: list[set[tuple[int, int]]],
        pred_sure: list[set[tuple[int, int]]],
        gold_possible: Optional[list[set[tuple[int, int]]]] = None,
        pred_possible: Optional[list[set[tuple[int, int]]]] = None,
    ) -> tuple[float, float, float, float]:
        assert len(gold_sure) == len(pred_sure), (
            f"Length mismatch: {len(gold_sure)} vs {len(pred_sure)}"
        )
        if gold_possible is not None:
            assert len(gold_possible) == len(gold_sure)
        if pred_possible is not None:
            assert len(pred_possible) == len(pred_sure)

        sum_a, sum_s = 0.0, 0.0
        sum_a_intersect_p, sum_a_intersect_s = 0.0, 0.0

        for i in range(len(gold_sure)):
            S = gold_sure[i]
            P = gold_possible[i] if gold_possible is not None else set()
            A_sure = pred_sure[i]
            A_possible = pred_possible[i] if pred_possible is not None else set()

            A = A_sure.union(A_possible)  # full prediction set
            sum_a += len(A)
            sum_s += len(S)
            sum_a_intersect_s += len(A & S)
            sum_a_intersect_p += len(A & (S | P))  # only matters if P exists

        if sum_a == 0:
            precision = 0.0
            recall = 0.0
            aer = 1.0
        else:
            precision = sum_a_intersect_p / sum_a
            recall = sum_a_intersect_s / sum_s if sum_s > 0 else 0.0
            aer = 1.0 - ((sum_a_intersect_s + sum_a_intersect_p) / (sum_s + sum_a))

        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return precision, recall, aer, f1_score
