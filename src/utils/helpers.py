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
    # we can simply take the number of unique ids - 1
    return torch.unique(word_ids_1d).shape[0] - 1


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
            padding_value=0,
        )

        # Determine true word counts from mappings (excluding padding)
        src_word_counts, tgt_word_counts = [], []
        src_token_lengths, tgt_token_lengths = [], []
        for b in batch:
            sw = b["source_token_to_word_mapping"]
            tw = b["target_token_to_word_mapping"]
            source_word_count = get_unique_words_from_mapping(sw)
            target_word_count = get_unique_words_from_mapping(tw)
            src_word_counts.append(source_word_count)
            tgt_word_counts.append(target_word_count)
            src_token_lengths.append(sw.shape[0])
            tgt_token_lengths.append(tw.shape[0])

        max_src_len = max(src_word_counts)
        max_tgt_len = max(tgt_word_counts)
        src_token_lengths = torch.tensor(src_token_lengths)
        tgt_token_lengths = torch.tensor(tgt_token_lengths)

        # Pad label matrix and label mask
        padded_labels = []
        label_masks = []
        for b in batch:
            label = b["label_matrix"]
            padded = torch.zeros((max_src_len, max_tgt_len), dtype=label.dtype)
            mask = torch.zeros((max_src_len, max_tgt_len), dtype=torch.bool)

            Si, Ti = label.shape
            padded[:Si, :Ti] = label
            mask[:Si, :Ti] = 1
            padded_labels.append(padded)
            label_masks.append(mask)

        labels = torch.stack(padded_labels)
        label_mask = torch.stack(label_masks)

        # this will allow the model to split the relevant parts for
        # source and target
        source_word_ids = pad_sequence(
            [b["source_token_to_word_mapping"].detach().clone() for b in batch],
            batch_first=True,
            padding_value=-1,
        )

        target_word_ids = pad_sequence(
            [b["target_token_to_word_mapping"].detach().clone() for b in batch],
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
            "source_token_lengths": src_token_lengths,
            "target_token_lengths": tgt_token_lengths,
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
