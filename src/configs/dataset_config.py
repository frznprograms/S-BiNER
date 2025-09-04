from dataclasses import dataclass
from typing import Callable, Optional

import torch

# from src.utils.helpers import collate_fn_span


@dataclass
class DatasetConfig:
    """
    Class to determine the configurations for downstream datsets. Mirrors
    the dataset configurations required for torch/transformers datasets.
    """

    source_lines_path: str
    target_lines_path: str
    alignments_path: str
    limit: Optional[int] = None
    one_indexed: bool = False
    context_sep: Optional[str] = " [WORD_SEP] "
    do_inference: bool = False
    log_output_dir: str = "logs"
    max_sentence_length: int = 512
    save: bool = False
    debug_mode: bool = False
    save_dir: str = "output"


@dataclass
class DataLoaderConfig:
    """
    Class to determine the configurations for downstram data loaders. Mirrors
    the configurations required for torch DataLoaders.
    """

    collate_fn: Optional[Callable] = None  # collate_fn_span
    batch_size: int = 4
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = not torch.backends.mps.is_available()  # Auto-disable for MPS
