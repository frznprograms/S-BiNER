from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, XLMRobertaPreTrainedModel
from src.configs.model_config import ModelConfig


@dataclass
class BinaryTokenClassificationModel(nn.Module):
    encoder: Union[RobertaPreTrainedModel, XLMRobertaPreTrainedModel]
    config: ModelConfig

    def __post_init__(self):
        super().__init__()
        # for [src || tgt] concatenation
        self.classifier = nn.Linear(self.config.hidden_size * 2, out_features=1)

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        source_mask: torch.BoolTensor,
        target_mask: torch.BoolTensor,
    ):
        """
        Arguments:
            input_ids: Tensor of shape (B, seq_len)
            attention_mask: Tensor of shape (B, seq_len)
            src_mask: Bool Tensor of shape (B, seq_len) with 1s at source positions
            tgt_mask: Bool Tensor of shape (B, seq_len) with 1s at target positions

        Returns:
            logits: Tensor of shape (B, src_len, tgt_len) with logits for each src-tgt token pair
        """
        # get the contextual token representations
        sequence_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, seq_len, H)
        # get source and target token embeddings using boolean masks
        source_repr = self.apply_mask(
            sequence_output, source_mask
        )  # (B, source_len, H)
        target_repr = self.apply_mask(
            sequence_output, target_mask
        )  # (B, target_len, H)

        B, S, H = source_repr.shape
        T = target_repr.shape[1]

        # create source-target pairwise combinations
        source_exp = source_repr.unsqueeze(2).expand(B, S, T, H)
        target_exp = target_repr.unsqueeze(2).expand(B, S, T, H)
        combined = torch.cat([source_exp, target_exp], dim=-1)  # (B, S, T, 2H)
        logits = self.classifier(combined).squeeze(-1)  # (B, S, T)

        return logits

    def apply_mask(self, sequence_output, mask: torch.BoolTensor):
        B, L, H = sequence_output.size()
        token_lists = []
        for i in range(B):
            indices = mask[i].nonzero(as_tuple=True)[0]
            token_lists.append(sequence_output[i, indices, :])

        return nn.utils.rnn.pad_sequence(token_lists, batch_first=True)


# @dataclass
# class SpanTokenAlignerOutput:
#     logits: Optional[torch.Tensor] = None
#     loss: Optional[torch.Tensor] = None


# class BinaryTokenClassification(nn.Module):
#     classifier: nn.Linear
#     dropout: nn.Dropout

#     def forward(
#         self,
#         model: Union[RobertaPreTrainedModel, XLMRobertaPreTrainedModel],
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#     ) -> SpanTokenAlignerOutput:
#         last_hidden_state: torch.Tensor = model(
#             input_ids, attention_mask=attention_mask
#         )[0]  # shape is (batch_size, seq_len, hidden_layer_dim=768)
#         logits = self.classifier(self.dropout(last_hidden_state)).to(torch.float32)

#         loss = None
#         if labels is not None:  # i.e. only for training, not inference
#             new_labels = torch.where(labels == -100, 0.0, labels)
#             # return loss for each individual token; no averaging yet:
#             loss_func = nn.BCEWithLogitsLoss(reduction="none")
#             loss = loss_func(logits.view(-1), new_labels.view(-1))
#             # zero out loss for ignored tokens:
#             loss = torch.where(labels.view(-1) == -100, 0, loss)

#             # average loss over valid tokens:
#             loss = torch.sum(loss) / (labels != -100).sum()

#         return SpanTokenAlignerOutput(loss=loss, logits=logits)
