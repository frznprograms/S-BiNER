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


@logger.catch(message="Unable to execute dataloader collate function", reraise=True)
def collate_fn_span(
    examples, tokenizer: PreTrainedTokenizer, do_inference: bool = True
):
    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

    # Flatten if necessary (for bidirectional samples)
    flat = []
    for ex in examples:
        if isinstance(ex, list):
            flat.extend(ex)
        else:
            flat.append(ex)

    # Convert to tensor before any use
    for i, x in enumerate(flat):
        x["input_ids"] = ensure_tensor(x["input_ids"])
        x["attention_mask"] = ensure_tensor(x["attention_mask"])
        x["labels"] = ensure_tensor(x["labels"])

        # print(f"[DEBUG] input_ids[{i}] shape: {x['input_ids'].shape}")

    # Now that all are tensors, we can safely pad
    input_ids = pad_sequence(
        [x["input_ids"] for x in flat],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,  # type:ignore
    )
    attention_mask = pad_sequence(
        [x["attention_mask"] for x in flat],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [x["labels"] for x in flat],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


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
