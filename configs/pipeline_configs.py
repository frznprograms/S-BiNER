from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    output_dir: Path
    log_dir: Optional[Path] = None
    save_checkpoint: bool = True
    batch_size: int = 32
    console_debug: bool = False

    def __post_init__(self):
        # Convert to Path objects if they aren't already
        self.output_dir = Path(self.output_dir)

        # Set default log_dir if None
        if self.log_dir is None:
            self.log_dir = Path("logs")

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
