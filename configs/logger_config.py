import sys
from datetime import datetime

from loguru import logger
from configs.pipeline_configs import PipelineConfig


class LoggedPipelineStep:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_logger()

    @logger.catch(message="Failed to create output directory", reraise=True)
    def make_output_directory(self):
        """Makes required output directories"""
        dirs = [
            self.config.output_dir,
            self.config.output_dir / "checkpoints",
            self.config.output_dir / "logs",
            self.config.output_dir / "final_output",
        ]

        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)

    @logger.catch(message="Logging setup failed", reraise=True)
    def setup_logger(self):
        """Setups the logging configuration for Loguru's logger"""

        timestamp = datetime.now().strftime("%d_%H%M")
        debug_dir = self.config.log_dir / "debug"  # type: ignore
        error_dir = self.config.log_dir / "error"  # type: ignore

        # Set min console level to DEBUG if true
        DEBUG = self.config.console_debug

        logger.remove()

        # Console Logger
        logger.add(
            sys.stderr,
            format="<green>{time: HH:mm:ss}</green> | <level>{level: <8}</level> | \
                <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG" if DEBUG else "INFO",
        )

        # File Logger (DEBUG)
        logger.add(
            debug_dir / f"debug_{timestamp}.log",
            format="{time: HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            retention="1 day",  # Max no of days to keep log files in logs directory
        )

        logger.add(
            error_dir / f"error_{timestamp}.log",
            format="{time: HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            backtrace=True,
            diagnose=True,
            retention="1 day",
        )

        logger.success(
            f"Logger initialized. Logs will be saved to {self.config.log_dir}"
        )

        return logger