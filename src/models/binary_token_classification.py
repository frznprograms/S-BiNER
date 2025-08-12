from typing import Callable, Union
import torch
import torch.nn as nn
from transformers import (
    RobertaPreTrainedModel,
    XLMRobertaPreTrainedModel,
)
from src.configs.model_config import ModelConfig
from loguru import logger

from src.utils.helpers import get_unique_words_from_mapping


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
        label_masks: torch.Tensor,
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
        combined_word_ids = torch.cat((source_word_ids, target_word_ids), dim=1)
        pooled_output_ids = self._pool_word_embeddings(
            outputs=outputs,
            batched_word_ids=combined_word_ids,
            attention_mask=attention_mask,
        )
        # combined_word_ids shape: (B, W_total)
        # pooled_output_ids shape: (B, W_total, H)
        print(f"Combined word ids shape: {combined_word_ids.shape}")

        # now that we have the pooled embeddings (which matches the number
        # of words in the sentence)
        pooled_src_embed, pooled_tgt_embed = [], []
        src_lengths, tgt_lengths = [], []
        for i in range(B):
            actual_no_of_src_words = get_unique_words_from_mapping(source_word_ids[i])
            actual_no_of_tgt_words = get_unique_words_from_mapping(target_word_ids[i])
            src_lengths.append(actual_no_of_src_words)
            tgt_lengths.append(actual_no_of_tgt_words)
            single_pooled_emb = pooled_output_ids[i]
            relevant_src_emb = single_pooled_emb[1:actual_no_of_src_words]
            relevant_tgt_emb = single_pooled_emb[
                actual_no_of_src_words + 1 : actual_no_of_tgt_words
            ]
            pooled_src_embed.append(relevant_src_emb)
            pooled_tgt_embed.append(relevant_tgt_emb)

        max_src_length = max(src_lengths)
        max_tgt_length = max(tgt_lengths)
        pooled_src_emb = (
            torch.tensor(pooled_src_embed)
            .unsqueeze(2)
            .expand(B, max_src_length, max_tgt_length)
        )
        pooled_tgt_emb = (
            torch.tensor(pooled_tgt_embed)
            .unsqueeze(1)
            .expand(B, max_src_length, max_tgt_length)
        )

        pairwise_classifier_inputs = torch.cat((pooled_src_emb, pooled_tgt_emb), dim=-1)
        print(f"Pairwise tensor shape: {pairwise_classifier_inputs.shape}")
        logits = self.classifier(self.dropout(pairwise_classifier_inputs)).squeeze(-1)

        # should the label mask even be utilised?

        print(f"Logits shape: {logits.shape}")

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
