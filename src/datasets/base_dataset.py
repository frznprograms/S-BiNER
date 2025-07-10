from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def read_data(self, path: str):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

    def save_data(self, data: Any, save_path: str, format: str = "csv") -> None:
        if format == "csv":
            data.to_csv(save_path)
        if format == "pt":
            torch.save(data, save_path)
