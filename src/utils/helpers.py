import os
import random
from typing import Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
import wandb
import yaml
from easydict import EasyDict
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer


def delist_the_list(items: list):
    for i in range(len(items)):
        items[i] = items[i][0]
    return items


def set_device(device_type: str = "auto") -> str:
    device = None
    if device_type == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    else:
        if device_type == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                logger.warning(
                    f"Could not set device to {device_type}, defaulting to cpu instead."
                )
                device = "cpu"
        if device_type == "mps":
            if torch.mps.is_available():
                device = "mps"
            else:
                logger.warning(
                    f"Could not set device to {device_type}, defaulting to cpu instead."
                )
                device = "cpu"
        else:
            device = "cpu"

    if device is None:
        raise ValueError(f"Could not assign device of type {device_type}")

    return device


@logger.catch(message="Unable to set seed for this run/experiment.", reraise=True)
def set_seeds(seed_num: Optional[int], deterministic: bool = True) -> int:
    if seed_num is None:
        logger.warning("A seed was not detected. Setting seed to 42.")
        seed_num = 42
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed_num


def get_unique_words_from_mapping(word_ids_1d: torch.Tensor) -> int:
    # since we know that special and padding tokens are always encoded as -1,
    # we can just ignore these when we count unique words
    valid = word_ids_1d[word_ids_1d >= 0]
    return int(valid.max().item() + 1) if valid.numel() else 0


# TODO: find out why we need to do .max().item() + 1 and what .numel() does
# TODO: look at dataset class - are special and padding tokens really encoded all as -1


def create_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(batch):
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,  # type: ignore
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )

        src_counts = [
            get_unique_words_from_mapping(b["source_token_to_word_mapping"])
            for b in batch
        ]
        tgt_counts = [
            get_unique_words_from_mapping(b["target_token_to_word_mapping"])
            for b in batch
        ]
        max_S = max(src_counts) if src_counts else 0
        max_T = max(tgt_counts) if tgt_counts else 0

        # pad labels + mask
        padded_labels, padded_masks = [], []
        for b in batch:
            L = b["label_matrix"]  # (S_i, T_i)
            S_i, T_i = L.shape
            P = torch.zeros((max_S, max_T), dtype=L.dtype)
            M = torch.zeros((max_S, max_T), dtype=torch.bool)
            P[:S_i, :T_i] = L
            M[:S_i, :T_i] = True
            padded_labels.append(P)
            padded_masks.append(M)

        labels = torch.stack(padded_labels)  # (B, max_S, max_T)
        label_mask = torch.stack(padded_masks)  # (B, max_S, max_T)

        source_word_ids = pad_sequence(
            [b["source_token_to_word_mapping"] for b in batch],
            batch_first=True,
            padding_value=-1,
        )
        target_word_ids = pad_sequence(
            [b["target_token_to_word_mapping"] for b in batch],
            batch_first=True,
            padding_value=-1,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_mask": label_mask,
            "source_word_ids": source_word_ids,
            "target_word_ids": target_word_ids,
        }

    return collate_fn


@logger.catch(message="Unable to parse an alignment.", reraise=True)
def parse_single_alignment(string, one_indexed=False, reverse: bool = False):
    assert "-" in string
    # just in case there are possible alignments denoted by p
    a, b = string.replace("p", "-").split("-")
    a, b = int(a), int(b)

    if one_indexed:
        a = a - 1
        b = b - 1

    if reverse:
        return b, a
    else:
        return a, b


def parse_config(
    config: Union[object, str, dict[str, Any]], config_class: type
) -> EasyDict:
    if isinstance(config, config_class):
        return EasyDict(config)
    elif isinstance(config, dict):
        return EasyDict(config_class(**config))
    elif isinstance(config, str):
        # Check if config is a file path
        if os.path.exists(config):
            with open(config, "r") as f:
                config_dict = yaml.safe_load(f)
            return EasyDict(config_class(**config_dict))
        else:
            # config is a YAML string
            config_dict = yaml.safe_load(config)
            return EasyDict(config_class(**config_dict))
    else:
        # For any other type, return empty EasyDict
        return EasyDict({})


def init_wandb_tracker():
    wandb.login()
    # wandb.init(project=self.project_name)


def init_tensorboard_tracker():
    raise NotImplementedError
