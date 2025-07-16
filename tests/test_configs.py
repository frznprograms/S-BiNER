from loguru import logger


def test_logger(get_logger):
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.success("This is a success message")


def test_configs(get_model_config, get_train_config, get_dataset_config):
    assert get_model_config.keys()
    assert get_train_config.keys()
    assert get_dataset_config.keys()
