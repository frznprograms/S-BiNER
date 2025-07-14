from src.models.binary_align import (
    XLMRobertaModelForBinaryTokenClassification,
    RobertaModelForBinaryTokenClassification,
)


class BinaryTokenClassificationFactory:
    def __init__(self, model_name_or_path: str, config):
        if "xlm" in model_name_or_path:
            return XLMRobertaModelForBinaryTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )
        else:
            return RobertaModelForBinaryTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )
