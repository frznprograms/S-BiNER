import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# Set min console level to DEBUG if true
DEBUG = True


class LoggedProcess:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.make_output_directory()  # Create directories first
        self.setup_logger()

    @logger.catch(message="Failed to create output directory", reraise=True)
    def make_output_directory(self):
        """Makes required output directories"""
        dirs = [
            self.output_dir,
            self.output_dir / "generic",
            self.output_dir / "debug",
            self.output_dir / "error",
        ]
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)

    @logger.catch(message="Logging setup failed", reraise=True)
    def setup_logger(self):
        timestamp = datetime.now().strftime("%d_%H%M")
        debug_dir = self.output_dir / "debug"
        error_dir = self.output_dir / "error"

        # Remove default logger
        logger.remove()

        # Console Logger
        logger.add(
            sys.stderr,
            format=self._get_colored_format(),
            level="DEBUG" if DEBUG else "INFO",
        )

        # File Logger (DEBUG)
        logger.add(
            debug_dir / f"debug_{timestamp}.log",
            format="{time: HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            retention="1 day",
        )

        # File Logger (ERROR)
        logger.add(
            error_dir / f"error_{timestamp}.log",
            format="{time: HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            backtrace=True,
            diagnose=True,
            retention="1 day",
        )

        logger.success(
            f"Logger initialized. Logs will be saved to {self.output_dir} \n"
        )
        return logger

    def _get_colored_format(self):  # Fixed method name
        """Returns format string with custom colors for different log levels"""

        def formatter(record):
            level = record["level"].name
            # Define colors for different levels
            colors = {
                "ERROR": "<red>",
                "WARNING": "<yellow>",
                "SUCCESS": "<green>",
                "INFO": "<white>",
                "DEBUG": "<cyan>",
            }
            level_color = colors.get(level, "<white>")
            return (
                "<green>{time: HH:mm:ss}</green> | "
                f"{level_color}{{level: <8}}</> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                f"{level_color}{{message}}</>"
            )

        return formatter


if __name__ == "__main__":
    log_process = LoggedProcess(output_dir="./test_logs")

    # Test different log levels
    logger.debug("This is a debug message")
    time.sleep(1)
    logger.info("This is an info message")
    time.sleep(1)
    logger.warning("This is a warning message")
    time.sleep(1)
    logger.error("This is an error message")
    time.sleep(1)
    logger.success("This is a success message")
    time.sleep(1)

    try:
        result = 1 / 0
    except Exception as e:
        logger.error(f"Division by zero error: {e}")
        logger.exception("Exception with full traceback:")
