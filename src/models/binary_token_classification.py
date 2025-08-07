from typing import Callable, Union
import torch
import torch.nn as nn
from transformers import (
    RobertaPreTrainedModel,
    XLMRobertaPreTrainedModel,
)
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

        print("-" * 50)
        print("Debugging how the model processes the tensors...")
        print("-" * 50)
        print(f"Input_id shapes: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Source word ids shape: {source_word_ids.shape}")
        print(f"Target word ids shape: {target_word_ids.shape}")
        print("-" * 50)
        print(f"Model outputs shape: {outputs.shape}")
        print("-" * 50)

        # pool token embeddings into word embeddings
        source_word_repr = self._pool_word_embeddings(
            outputs=outputs,
            batched_word_ids=source_word_ids,
            attention_mask=attention_mask,
        )
        target_word_repr = self._pool_word_embeddings(
            outputs=outputs,
            batched_word_ids=target_word_ids,
            attention_mask=attention_mask,
        )
        print(
            f"Source word representation shapes (after word pooling): {source_word_repr.shape}"
        )
        print(
            f"Target word representation shapes (after word pooling): {target_word_repr.shape}"
        )
        print("-" * 50)

        S = source_word_repr.shape[1]  # source_word_repr has shape (B, S, H)
        T = target_word_repr.shape[1]

        source_exp = source_word_repr.unsqueeze(2)  # (B, S, 1, H)
        source_exp = source_exp.expand(B, S, T, H)
        target_exp = target_word_repr.unsqueeze(1)  # (B, 1, T, H)
        target_exp = target_exp.expand(B, S, T, H)

        combined = torch.cat([source_exp, target_exp], dim=-1)  # (B, S, T, 2H)
        print(f"Combined logits shape: {combined.shape}")
        logits = self.classifier(self.dropout(combined)).squeeze(-1)
        # (B, S, T, 1) -> (B, S, T)
        return logits

    @logger.catch(message="Unable to pool embeddings properly", reraise=True)
    def _pool_word_embeddings(
        self,
        outputs: torch.Tensor,  # (B, L, H)
        batched_word_ids: torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        B, L, H = outputs.shape
        pooled_embeddings = []

        for i in range(B):
            output = outputs[i]  # (L, H)
            word_ids = batched_word_ids[i]  # (L,)
            mask = attention_mask[i]  # (L,)

            word_embeddings = []
            current_word_id = None
            current_vectors = []

            for j in range(L):
                if mask[j] == 0:
                    continue  # Skip padding

                if word_ids[j] == -1:
                    # Special token – treat as its own pooled embedding
                    word_embeddings.append(output[j])
                    continue

                if word_ids[j] != current_word_id:
                    # start a new word span
                    if current_vectors:
                        word_embeddings.append(
                            agg_fn(torch.stack(current_vectors), dim=0)
                        )
                    current_vectors = [output[j]]
                    current_word_id = word_ids[j]
                else:
                    current_vectors.append(output[j])

            # Final word
            if current_vectors:
                word_embeddings.append(agg_fn(torch.stack(current_vectors), dim=0))

            pooled_embeddings.append(torch.stack(word_embeddings))

        return nn.utils.rnn.pad_sequence(pooled_embeddings, batch_first=True)

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
