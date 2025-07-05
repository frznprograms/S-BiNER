import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from configs.pipeline_configs import PipelineConfig


class PipelineStep(ABC):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.step_name = self.__class__.__name__

    @abstractmethod
    def execute(self):
        pass

    def save_checkpoint(self, data: Any, checkpoint_path: Path):
        if self.config.save_checkpoint:
            if isinstance(data, pd.DataFrame):
                data.to_csv(checkpoint_path)
            else:
                with open(checkpoint_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
