import json
from multiprocessing import cpu_count
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import wandb
import yaml
from easydict import EasyDict
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers.modeling_utils import PreTrainedModel


def write_hf_checkpoint(
    model: PreTrainedModel,
    save_dir: Union[str, Path],
    config_dict: Dict[str, Any],
    safe_serialization: bool = True,
):
    save_dir = Path(save_dir)  # type: ignore
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "custom_config.json", "w") as f:  # type: ignore
        json.dump(config_dict, f, indent=4)

    state = model.state_dict()
    if safe_serialization:
        save_file(state, str(save_dir / "model.safetensors"))  # type: ignore
    else:
        torch.save(state, save_dir / "pytorch_model.bin")  # type: ignore


def load_hf_checkpoint(load_dir: Union[str, Path], map_location: str = "cpu"):
    load_dir = Path(load_dir)  # type: ignore
    st_path_safe = load_dir / "model.safetensors"  # type: ignore
    st_path_torch = load_dir / "pytorch_model.bin"  # type: ignore

    if st_path_safe.exists():
        return load_file(str(st_path_safe), device=map_location)
    elif st_path_torch.exists():
        return torch.load(st_path_torch, map_location=map_location)
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {load_dir}"
        )


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


def get_num_workers(default_max: int = 3) -> int:
    try:
        system_cores = cpu_count()
        capped_cores = min(max(1, system_cores), default_max)
        user_defined = os.environ.get("PIGEON_NUM_WORKERS")
        return int(user_defined) if user_defined else capped_cores
    except Exception:
        return 1  # just use single subprocess


def init_wandb_tracker():
    wandb.login()
    # wandb.init(project=self.project_name)


def init_tensorboard_tracker():
    raise NotImplementedError
