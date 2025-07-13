import torch
import torch.nn as nn
# import torch.nn.functional as F 
from dataclasses import dataclass
from typing import Optional
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)


@dataclass
class SpanTokenAlignerOutput:
    logits: Optional[torch.FloatTensor]
    loss: Optional[torch.FloatTensor]


class AutoModelForBinaryTokenClassification:
    def __init__(self, model_name_or_path: str, config):
        if "xlm" in model_name_or_path:
            return XLMRobertaModelForTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )
        else:
            return RobertaModelForTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )


class BinaryTokenClassification(nn.Module):
    pass


class RobertaModelForTokenClassification(
    RobertaPreTrainedModel, BinaryTokenClassification
):
    def __init__(self, config):
        self.config = config
        self.model = RobertaModel(config=config, add_pooling_layer=False)


class XLMRobertaModelForTokenClassification(
    XLMRobertaPreTrainedModel, BinaryTokenClassification
):
    def __init__(self, config):
        self.config = config
        self.model = XLMRobertaModel(config=config, add_pooling_layer=False)
