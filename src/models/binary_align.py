import torch
import torch.nn as nn

# import torch.nn.functional as F
from typing import Optional
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)
from src.models.binary_token_classification import (
    BinaryTokenClassification,
    SpanTokenAlignerOutput,
)


# TODO: find out if it is better to use RobertaModelForTokenClassification
# TODO: ensure correct configuration set up for each model
# -> num_labels, classifier_dropout, hidden_dropout_prob, hidden_size etc.
# TODO: document shape tracking as tensors move through the model


class RobertaModelForBinaryTokenClassification(
    RobertaPreTrainedModel, BinaryTokenClassification
):
    def __init__(self, config):
        super().__init__(config=config)
        self.num_labels = config.num_labels
        # prevent pooling to get token-level outputs, not sentence-level
        self.model = RobertaModel(config=config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=1)

        # initialise weights and final pre-processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SpanTokenAlignerOutput:
        return super(RobertaModelForBinaryTokenClassification, self).forward(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


class XLMRobertaModelForBinaryTokenClassification(
    XLMRobertaPreTrainedModel, BinaryTokenClassification
):
    def __init__(self, config):
        super().__init__(config=config)
        self.num_labels = config.num_labels
        # prevent pooling to get token-level outputs, not sentence-level
        self.model = XLMRobertaModel(config=config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=1)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SpanTokenAlignerOutput:
        return super(XLMRobertaModelForBinaryTokenClassification, self).forward(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
