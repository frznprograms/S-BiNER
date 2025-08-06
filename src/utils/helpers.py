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


@logger.catch(message="Model unable to collate data", reraise=True)
def create_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(batch):
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,  # type: ignore
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0.0,
        )

        # Determine max dimensions for label matrix
        max_src_len = max(b["label_matrix"].shape[0] for b in batch)
        max_tgt_len = max(b["label_matrix"].shape[1] for b in batch)

        # Prepare padded labels and label masks
        padded_labels = []
        label_masks = []
        for b in batch:
            label = b["label_matrix"]
            padded = torch.zeros((max_src_len, max_tgt_len), dtype=label.dtype)
            padded[: label.shape[0], : label.shape[1]] = label
            padded_labels.append(padded)

            mask = torch.zeros((max_src_len, max_tgt_len), dtype=torch.bool)
            mask[: label.shape[0], : label.shape[1]] = 1
            label_masks.append(mask)

        labels = torch.stack(padded_labels)
        label_mask = torch.stack(label_masks)

        source_word_ids = [
            torch.tensor(b["source_token_to_word_mapping"]) for b in batch
        ]
        target_word_ids = [
            torch.tensor(b["target_token_to_word_mapping"]) for b in batch
        ]

        source_word_ids = torch.nn.utils.rnn.pad_sequence(
            source_word_ids, batch_first=True, padding_value=-1
        )
        target_word_ids = torch.nn.utils.rnn.pad_sequence(
            target_word_ids, batch_first=True, padding_value=-1
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
