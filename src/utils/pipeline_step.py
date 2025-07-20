from abc import ABC, abstractmethod
from pathlib import Path


class PipelineStep(ABC):
    checkpoint_dir: Path = Path("/checkpoints")
    debug_mode: bool = True

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.step_name = self.__class__.__name__

    @abstractmethod
    def run(self):
        pass
