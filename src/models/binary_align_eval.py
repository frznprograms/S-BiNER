from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Union

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
    ):
        # TODO: find out if need to use partial()
        return self._bidirectional_eval_span(
            dataloader=dataloader,
            model=model,
            threshold=threshold,
            sure=sure,
            device=device,
            mini_batch_size=mini_batch_size,
            bidirectional_combine_type=bidirectional_combine_type,
            tk2word_prob=tk2word_prob,
        )

    @logger.catch(message="Unable to get bidirectional metrics", reraise=True)
    @torch.no_grad
    def _bidirectional_eval_span(
        self,
        dataloader: DataLoader,
        model,
        threshold: float,
        sure: list[set[tuple[int, int]]],
        device: str,
        mini_batch_size: int,
        bidirectional_combine_type: str = "intersection",
        tk2word_prob: str = "mean",
    ) -> tuple[float, float, float, float]:
        model.eval()

        predicted_sure = []

        # note that each sample is actually a batch of batch_size
        for sample in tqdm(dataloader):
            if isinstance(sample, list):  # Bidirectional eval
                sample_1, sample_2 = sample
                sample_preds_1 = self._get_sample_probs(
                    sample=sample_1,
                    model=model,
                    device=device,
                    mini_batch_size=mini_batch_size,
                    tk2word_prob=tk2word_prob,
                )
                sample_preds_2 = self._get_sample_probs(
                    sample=sample_2,
                    model=model,
                    device=device,
                    mini_batch_size=mini_batch_size,
                    tk2word_prob=tk2word_prob,
                    reverse=True,
                )
                all_probs = self._combine_pred_probs(
                    preds_1=sample_preds_1,
                    preds_2=sample_preds_2,
                    method=bidirectional_combine_type,
                )
            else:  # Unidirectional
                all_probs = self._get_sample_probs(
                    sample, model, device, mini_batch_size, tk2word_prob
                )

            sure_set = {k for k, v in all_probs.items() if v >= threshold}
            predicted_sure.append(sure_set)

        metrics = self.calculate_metrics(
            gold_sure=sure,
            pred_sure=predicted_sure,
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
        # access index 0 to remove wrapper dimension: (1, rows, cols) -> (rows, cols)
        all_input_ids = sample["input_ids"][0].split(mini_batch_size)
        all_attention_mask = sample["attention_mask"][0].split(mini_batch_size)
        bpe2word_map = sample["bpe2word_map"][0].split(mini_batch_size)

        # BinaryAlignEvaluator.view_input_ids(
        #     sample=sample, mini_batch_size=mini_batch_size
        # )
        count = 0

        for input_ids, attention_mask, bpe2word_batch in zip(
            all_input_ids, all_attention_mask, bpe2word_map
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            if logits.shape[2] == 2:
                probs = softmax(logits)[:, :, 1]
            else:
                probs = sigmoid(logits)[:, :, 0]

            for batch_idx in range(probs.size(0)):
                sentence_probs = probs[batch_idx]

                # account for 1D case
                if bpe2word_batch.dim() == 1:
                    bpe2word = bpe2word_batch.tolist()
                else:
                    bpe2word = bpe2word_batch[batch_idx].tolist()

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
    def calculate_metrics(self, gold_sure, pred_sure):
        sum_a = 0.0  # total predicted alignments
        sum_s = 0.0  # total gold (sure) alignments
        sum_a_intersect_s = 0.0  # correctly predicted alignments

        for S, A in zip(gold_sure, pred_sure):
            sum_a += len(A)
            sum_s += len(S)
            sum_a_intersect_s += len(A & S)

        precision = sum_a_intersect_s / sum_a if sum_a != 0 else 0.0
        recall = sum_a_intersect_s / sum_s if sum_s != 0 else 0.0

        # FIXED: Correct AER calculation
        # AER = (|A| + |S| - 2 * |A ∩ S|) / (|A| + |S|)
        aer_denom = sum_a + sum_s
        if aer_denom > 0:
            aer = (sum_a + sum_s - 2 * sum_a_intersect_s) / aer_denom
        else:
            aer = 1.0  # No predictions and no gold alignments = perfect error rate

        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        # Add debug information
        logger.debug("  Metrics calculation:")
        logger.debug(f"  Total predicted alignments: {sum_a}")
        logger.debug(f"  Total gold alignments: {sum_s}")
        logger.debug(f"  Correct predictions: {sum_a_intersect_s}")
        logger.debug(f"  Precision: {precision:.4f}")
        logger.debug(f"  Recall: {recall:.4f}")
        logger.debug(f"  AER: {aer:.4f}")
        logger.debug(f"  F1: {f1:.4f}")

        return precision, recall, aer, f1

    @staticmethod
    def view_input_ids(sample, mini_batch_size: int) -> None:
        print("=" * 50)
        print("Summary:")
        print(
            f"Input ids is a {type(sample['input_ids'])} of shape {sample['input_ids'].shape}"
        )
        print(
            f"Input ids element is a {type(sample['input_ids'][0])} of shape {sample['input_ids'][0].shape}"
        )
        print("=" * 50)
        print(
            f"Attention mask is a {type(sample['attention_mask'])} of shape {sample['attention_mask'].shape}"
        )
        print(
            f"Attention mask element is a {type(sample['attention_mask'][0])} of shape {sample['attention_mask'][0].shape}"
        )
        print("=" * 50)
        print(
            f"bpe2word mapping is a {type(sample['bpe2word_map'])} of shape {sample['bpe2word_map'].shape}"
        )
        print(
            f"bpe2word mapping element is a {type(sample['bpe2word_map'][0])} of shape {sample['bpe2word_map'][0].shape}"
        )
        print("=" * 50)
        print("Inputs before splitting into mini batches:")
        print(f"Input ids: {sample['input_ids'][0]}")
        print(f"Attention mask: {sample['attention_mask'][0]}")
        print(f"bpe2word mapping: {sample['bpe2word_map']}")
        print("=" * 50)
        print("Inputs after splitting into mini batches:")
        print(f"Input ids: {sample['input_ids'][0].split(mini_batch_size)}")
        print(f"Attention mask: {sample['attention_mask'][0].split(mini_batch_size)}")
        print(f"bpe2word mapping: {sample['bpe2word_map'][0].split(mini_batch_size)}")
        print("=" * 50)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from src.configs.model_config import ModelConfig
    from src.configs.dataset_config import DatasetConfig, DataLoaderConfig
    from src.configs.train_config import TrainConfig
    from src.datasets.datasets_silver import AlignmentDatasetSilver
    from src.models.binary_align_trainer import BinaryAlignTrainer
    from src.utils.helpers import collate_fn_span

    model_config = ModelConfig(model_name_or_path="FacebookAI/roberta-base")
    train_config = TrainConfig(experiment_name="trainer-test", mixed_precision="no")
    train_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/train.src",
        target_lines_path="data/cleaned_data/train.tgt",
        alignments_path="data/cleaned_data/train.talp",
        limit=4,
    )
    eval_dataset_config = DatasetConfig(
        source_lines_path="data/cleaned_data/dev.src",
        target_lines_path="data/cleaned_data/dev.tgt",
        alignments_path="data/cleaned_data/dev.talp",
        limit=2,
        do_inference=True,
    )
    dataloader_config = DataLoaderConfig(collate_fn=collate_fn_span)
    tok = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    train_data = AlignmentDatasetSilver(tokenizer=tok, **train_dataset_config.__dict__)
    eval_data = AlignmentDatasetSilver(tokenizer=tok, **eval_dataset_config.__dict__)

    trainer = BinaryAlignTrainer(
        tokenizer=tok,
        model_config=model_config,
        train_config=train_config,
        dataset_config=train_dataset_config,
        dataloader_config=dataloader_config,
        train_data=train_data,
        eval_data=eval_data,
        seed_num=1,
    )
    trainer.run()
