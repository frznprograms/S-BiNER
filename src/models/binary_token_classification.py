from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, XLMRobertaPreTrainedModel


@dataclass
class SpanTokenAlignerOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


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
