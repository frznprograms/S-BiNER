from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PipelineStep(ABC):
    checkpoint_dir: Path = Path("/checkpoints")
    console_debug: bool = True

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.step_name = self.__class__.__name__

    @abstractmethod
    def run(self):
        pass

    def save_checkpoint(self, data: Any, checkpoint_path: Path):
        # if self.config.save_checkpoint:
        #     if isinstance(data, pd.DataFrame):
        #         data.to_csv(checkpoint_path)
        #     else:
        #         with open(checkpoint_path, "w") as f:
        #             json.dump(data, f, indent=4, ensure_ascii=False)

        """Save model parameters?"""
        pass
