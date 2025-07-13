import torch
import torch.nn as nn

# import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Union
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)


@dataclass
class SpanTokenAlignerOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class AutoModelForBinaryTokenClassification:
    def __init__(self, model_name_or_path: str, config):
        if "xlm" in model_name_or_path:
            return XLMRobertaModelForBinaryTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )
        else:
            return RobertaModelForBinaryTokenClassification.from_pretrained(
                model_name_or_path, config=config
            )


class BinaryTokenClassification(nn.Module):
    classifier: nn.Linear
    dropout: nn.Dropout

    def forward(
        self,
        model: Union[RobertaPreTrainedModel, XLMRobertaPreTrainedModel],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SpanTokenAlignerOutput:
        last_hidden_state: torch.Tensor = model(
            input_ids, attention_mask=attention_mask
        )[0]  # shape is (batch_size, seq_len, hidden_layer_dim=768)
        logits = self.classifier(self.dropout(last_hidden_state)).to(torch.float32)

        loss = None
        if labels is not None:  # i.e. only for training, not inference
            new_labels = torch.where(labels == -100, 0.0, labels)
            # return loss for each individual token; no averaging yet:
            loss_func = nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_func(logits.view(-1), new_labels.view(-1))
            # zero out loss for ignored tokens:
            loss = torch.where(labels.view(-1) == -100, 0, loss)
            # average loss over valid tokens:
            loss = torch.sum(loss) / (labels != -100).sum()

        return SpanTokenAlignerOutput(loss=loss, logits=logits)


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
