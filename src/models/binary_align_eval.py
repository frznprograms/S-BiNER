from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
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
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
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
        model.eval()
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
        tk2word_prob: str = "max",
    ) -> tuple[float, float, float, float]:
        model.eval()

        predicted_sure = []
        # note that each sample is actually a batch of batch_size
        for sample in tqdm(dataloader):
            if self.debug_mode:
                self.view_model_structure(sample=sample, model=model, device=device)
                break
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
                    threshold=threshold,
                )
            else:  # Unidirectional
                all_probs = self._get_sample_probs(
                    sample, model, device, mini_batch_size, tk2word_prob
                )

            sure_set = {k for k, v in all_probs.items() if v >= threshold}  # type: ignore
            predicted_sure.append(sure_set)

            if self.debug_mode:
                self.view_gold_alignments(sure)
                self.view_all_probs(all_probs=all_probs, threshold=threshold)

        metrics = self.calculate_metrics_sure_only(
            gold_sure_alignments=sure,
            predicted_sure_alignments=predicted_sure,
        )

        # set model back to training mode
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
        bpe2word_map = sample["bpe2word_map"].tolist()

        if self.debug_mode:
            self.view_input_ids(sample=sample, mini_batch_size=mini_batch_size)

        global_sentence_idx = 0  # Global counter across all mini-batches

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

            for batch_idx, sentence_probs in enumerate(probs):
                current_sentence_idx = global_sentence_idx + batch_idx

                for word_number, word_prob in enumerate(sentence_probs):
                    if word_number < len(bpe2word_batch):
                        target = bpe2word_batch[word_number]

                        if target != -1:
                            target = int(target)

                            if reverse:
                                sample_preds[(target, current_sentence_idx)].append(
                                    word_prob.item()
                                )
                            else:
                                sample_preds[(current_sentence_idx, target)].append(
                                    word_prob.item()
                                )

            global_sentence_idx += input_ids.size(0)

        result = {k: AGG_FN[tk2word_prob](v) for k, v in sample_preds.items()}
        # print(
        #     f"Debug _get_sample_probs: returning {len(result)} items, sample: {list(result.items())[-5:-1]}"
        # )
        return result

    @logger.catch(
        message="Unable to combine prediction probabilities for a sample", reraise=True
    )
    def _combine_pred_probs(
        self,
        preds_1: dict[tuple[int, int], float],
        preds_2: dict[tuple[int, int], float],
        threshold: float,
        method: str,
    ) -> set[tuple[int, int]]:
        current_preds = set()
        preds_1_set = set(k for k, v in preds_1.items() if v > threshold)
        preds_2_set = set(k for k, v in preds_2.items() if v > threshold)
        if method == "union":
            current_preds = preds_1_set.union(preds_2_set)
        elif method == "intersection":
            current_preds = preds_1_set.intersection(preds_2_set)
        elif method == "avg" or method == "bidi_avg":
            current_preds_with_probs = {
                k: (preds_1.get(k, 0) + preds_2.get(k, 0)) / 2
                for k in preds_1_set | preds_2_set
            }
            current_preds = set(
                {k: v for k, v in current_preds_with_probs.items() if v > threshold}
            )
        else:
            logger.error("Unknown bidirectional combine method.")

        return current_preds

    @logger.catch(message="Unable to calculate metrics", reraise=True)
    def calculate_metrics_sure_only(
        self, gold_sure_alignments, predicted_sure_alignments
    ):
        assert len(gold_sure_alignments) == len(predicted_sure_alignments), (
            f"Mismatch in number of sentence pairs: {len(gold_sure_alignments)} vs {len(predicted_sure_alignments)}"
        )
        # print(f"Debug: Gold alignments sample: {list(gold_sure_alignments)[:3]}")
        # print(
        #     f"Debug: Predicted alignments sample: {list(predicted_sure_alignments)[:3]}"
        # )

        sum_a_intersect_s, sum_s, sum_a = 0.0, 0.0, 0.0

        for S, A in zip(gold_sure_alignments, predicted_sure_alignments):
            sum_a += len(A)  # Total predicted alignments
            sum_s += len(S)  # Total sure alignments (ground truth)
            sum_a_intersect_s += len(A.intersection(S))  # Correct predictions

        # Calculate metrics
        precision = sum_a_intersect_s / sum_a if sum_a != 0 else 0.0
        recall = sum_a_intersect_s / sum_s if sum_s != 0 else 0.0

        # AER (Alignment Error Rate) - simplified for sure alignments only
        aer = (
            1.0 - (2 * sum_a_intersect_s) / (sum_a + sum_s)
            if (sum_a + sum_s) != 0
            else 1.0
        )

        # F1 Score
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Add debug information
        # logger.debug("Metrics calculation:")
        # logger.debug(f"  Total predicted alignments: {sum_a}")
        # logger.debug(f"  Total gold alignments: {sum_s}")
        # logger.debug(f"  Correct predictions: {sum_a_intersect_s}")
        # logger.debug(f"  Precision: {precision:.4f}")
        # logger.debug(f"  Recall: {recall:.4f}")
        # logger.debug(f"  AER: {aer:.4f}")
        # logger.debug(f"  F1: {f1_score:.4f}")

        return precision, recall, aer, f1_score

    def view_input_ids(self, sample, mini_batch_size: int) -> None:
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
        print(f"bpe2word mapping: {sample['bpe2word_map'][0].tolist()}")
        print("=" * 50)

    def view_model_structure(self, sample, model, device):
        """Debug function to understand what the model is actually predicting"""

        print("=== MODEL STRUCTURE ANALYSIS ===")

        # Get a small batch to analyze
        input_ids = sample["input_ids"][0][:1].to(device)  # Just first sentence
        attention_mask = sample["attention_mask"][0][:1].to(device)

        print(f"Input shape: {input_ids.shape}")
        print(f"Sample input_ids: {input_ids[0][:20]}...")  # First 20 tokens

        # Get model output
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        print(f"Logits shape: {logits.shape}")
        print("Expected: [batch_size, sequence_length, num_classes]")
        print(f"Actual: [{logits.shape[0]}, {logits.shape[1]}, {logits.shape[2]}]")

        # Check what the model is classifying
        if logits.shape[2] == 1:
            print("Model output: Single value per token (binary classification)")
            print("This suggests: For each token, predict if it's aligned (binary)")
        elif logits.shape[2] == 2:
            print(
                "Model output: Two values per token (binary classification with both classes)"
            )
            print(
                "This suggests: For each token, predict [not_aligned, aligned] probabilities"
            )
        else:
            print(f"Model output: {logits.shape[2]} classes per token")

        # Check the actual values
        if logits.shape[2] == 2:
            probs = torch.softmax(logits, dim=-1)
            print("Sample probabilities (first 10 tokens):")
            for i in range(min(10, logits.shape[1])):
                print(
                    f"  Token {i}: not_aligned={probs[0, i, 0]:.6f}, aligned={probs[0, i, 1]:.6f}"
                )
        else:
            probs = torch.sigmoid(logits)
            print("Sample probabilities (last 10 tokens):")
            start_idx = max(0, logits.shape[1] - 10)
            for i in range(start_idx, logits.shape[1]):
                print(f"  Token {i}: aligned_prob={probs[0, i, 0]:.6f}")

        # Check bpe2word mapping
        bpe2word = sample["bpe2word_map"][0].tolist()
        print(f"\nBPE to word mapping (first 20): {bpe2word[:20]}")
        print(f"Unique word IDs: {sorted(set([x for x in bpe2word if x != -1]))}")

        return logits, probs

    def view_gold_alignments(self, sure_alignments) -> None:
        """Analyze the structure of gold alignments"""
        print("=== GOLD ALIGNMENTS ANALYSIS ===")

        for i, alignment_set in enumerate(sure_alignments[:3]):
            print(f"\nSentence pair {i}:")
            print(f"  Number of alignments: {len(alignment_set)}")
            print(f"  Sample alignments: {list(alignment_set)[:5]}")

            # Analyze the pattern
            sources = [pair[0] for pair in alignment_set]
            targets = [pair[1] for pair in alignment_set]

            print(f"  Source word range: {min(sources)} to {max(sources)}")
            print(f"  Target word range: {min(targets)} to {max(targets)}")
            print(f"  Unique source words: {len(set(sources))}")
            print(f"  Unique target words: {len(set(targets))}")

    def view_all_probs(self, all_probs, threshold: float) -> None:
        print(f"Debug: all_probs type: {type(all_probs)}")
        print(f"Debug: all_probs length: {len(all_probs)}")
        if len(all_probs) > 0:
            sample_items = list(all_probs.items())[:5]  # type: ignore # First 5 items
            print(f"Debug: sample all_probs items: {sample_items}")
            prob_values = list(all_probs.values())  # type: ignore
            print(
                f"Debug: prob range: min={min(prob_values):.4f}, max={max(prob_values):.4f}"
            )
        print(f"Debug: threshold: {threshold}")
        print(
            f"Debug: items above threshold: {[(k, v) for k, v in all_probs.items() if v >= threshold][:5]}"  # type: ignore
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from src.configs.dataset_config import DataLoaderConfig, DatasetConfig
    from src.configs.model_config import ModelConfig
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
