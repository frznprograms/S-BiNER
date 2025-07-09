from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseDataset(ABC):
    def __len__(self):
        pass

    @abstractmethod
    def read_data(self, path: Path):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def save_data(self, save_path: Path):
        pass
