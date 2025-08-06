from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Union
import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, XLMRobertaPreTrainedModel
from src.configs.model_config import ModelConfig
from loguru import logger


class BinaryTokenClassificationModel(nn.Module):
    def __init__(
        self,
        encoder: Union[RobertaPreTrainedModel, XLMRobertaPreTrainedModel],
        config: ModelConfig,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size * 2, out_features=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_word_ids: torch.Tensor,
        target_word_ids: torch.Tensor,
    ):
        # note that input_ids have shape (B, L)
        # get token-level hidden states
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, L, H)
        B, L, H = outputs.shape

        # pool token embeddings into word embeddings
        source_word_repr = self._pool_word_embeddings(outputs, source_word_ids)
        target_word_repr = self._pool_word_embeddings(outputs, target_word_ids)

        S = source_word_repr.shape[1]  # source_word_repr has shape (B, S, H)
        T = target_word_repr.shape[1]

        source_exp = source_word_repr.unsqueeze(2)  # (B, S, 1, H)
        source_exp = source_exp.expand(B, S, T, H)
        target_exp = target_word_repr.unsqueeze(1)  # (B, 1, T, H)
        target_exp = target_exp.expand(B, S, T, H)

        combined = torch.cat([source_exp, target_exp], dim=-1)  # (B, S, T, 2H)
        logits = self.classifier(
            self.dropout(combined).squeeze(-1)
        )  # (B, S, T, 1) -> (B, S, T)

        return logits

    @logger.catch(message="Model unable to pool embeddings", reraise=True)
    def _pool_word_embeddings(
        self, outputs: torch.Tensor, batched_word_ids: torch.Tensor
    ) -> torch.Tensor:
        """pools token embeddings to their corresponding word level"""
        B, L, H = outputs.size()
        pooled = []
        for i in range(B):
            word_vectors = []
            current_word_id = None
            current_vecs = []
            for j, word_id in enumerate(batched_word_ids[i]):
                if word_id is None:
                    continue  # ignore separator tokens
                if word_id != current_word_id:
                    if current_vecs:
                        word_vectors.append(torch.stack(current_vecs).mean(dim=0))
                        # TODO: here we use mean pooling, let it be customizable
                    current_vecs = []
                    current_word_id = word_id
                current_vecs.append(outputs[i, j])
            if current_vecs:
                word_vectors.append(torch.stack(current_vecs).mean(dim=0))
            if word_vectors:
                pooled.append(torch.stack(word_vectors))
            else:
                pooled.append(torch.zeros(1, H, device=outputs.device))

        return nn.utils.rnn.pad_sequence(pooled, batch_first=True)

    # TODO: if keeping this here, remove from src.helpers
    @logger.catch(message="Model unable to collate data", reraise=True)
    def create_collate_fn(self, tokenizer: PreTrainedTokenizer):
        def collate_fn(batch):
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"] for b in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,  # type: ignore
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [b["attention_mask"] for b in batch],
                batch_first=True,
                padding_value=0.0,
            )
            labels = torch.stack([b["label_matrix"] for b in batch])
            source_word_ids = [b["source_token_to_word_mapping"] for b in batch]
            target_word_ids = [b["target_token_to_word_mapping"] for b in batch]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "source_word_ids": source_word_ids,
                "target_word_ids": target_word_ids,
            }

        return collate_fn

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
