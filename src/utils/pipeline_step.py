from abc import ABC, abstractmethod
from pathlib import Path
import wandb


class PipelineStep(ABC):
    checkpoint_dir: Path = Path("checkpoints")
    debug_mode: bool = False
    project_name: str = "binary-align-for-zh-ner"

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.step_name = self.__class__.__name__

    @abstractmethod
    def run(self):
        pass

    def _init_wandb_tracker(self):
        wandb.login()
        # wandb.init(project=self.project_name)
