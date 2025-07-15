import random
from typing import Optional

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
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


def set_seeds(seed_num: Optional[int], deterministic: bool = True) -> None:
    if seed_num is None:
        seed_num = 42
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_fn_span(examples: list[dict], tokenizer: PreTrainedTokenizer):
    def _get_examples(examples):
        example1 = []
        example2 = []
        for example in examples:
            example1.append(example[0])
            example2.append(example[1])
        return example1, example2

    def _produce_batch(examples):
        input_ids = [torch.tensor(x["input_ids"]) for x in examples]
        attention_mask = [torch.tensor(x["attention_mask"]) for x in examples]
        labels = [torch.tensor(x["labels"]) for x in examples]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,  # type:ignore
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    if isinstance(examples[0], tuple):
        examples1, examples2 = _get_examples(examples)
        batch1 = _produce_batch(examples1)
        batch2 = _produce_batch(examples2)
    else:
        return _produce_batch(examples)

    return {
        **{f"{key}1": value for key, value in batch1.items()},
        **{f"{key}2": value for key, value in batch2.items()},
    }
