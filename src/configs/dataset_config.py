from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig(ABC):
    source_lines_path: str
    target_lines_path: str
    alignments_path: str
    limit: Optional[int] = None
    one_indexed: bool = False
    context_sep: Optional[str] = " [WORD_SEP] "
    do_inference: bool = False
    log_output_dir: str = "logs"
    data_type: str = "silver"  # "silver" or "gold"
    save: bool = False
    batch_size: int = 16
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
