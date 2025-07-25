import torch
import torch.nn as nn

# import torch.nn.functional as F
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from src.models.binary_token_classification import BinaryTokenClassificationModel
from src.configs.model_config import ModelConfig

from loguru import logger


class RobertaModelForBinaryTokenClassification(
    RobertaPreTrainedModel, BinaryTokenClassificationModel
):
    def __init__(self, config: ModelConfig):
        roberta_config = config._to_roberta_config()
        super().__init__(config=roberta_config)
        self.num_labels = self.config.num_labels

        # prevent pooling to get token-level, not sentence level outputs
        self.encoder = RobertaModel(config=roberta_config, add_pooling_layer=False)
        self._add_special_tokens(tokenizer=self.encoder.config.tokenizer_class)

        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

        # initialise weights and final processing
        self.__post_init__()

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        source_mask: torch.BoolTensor,
        target_mask: torch.BoolTensor,
    ):
        return super(RobertaModelForBinaryTokenClassification, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_mask=source_mask,
            target_mask=target_mask,
        )

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def _add_special_tokens(self, tokenizer: PreTrainedTokenizer):
        tokenizer_vocab_size = len(tokenizer)
        curr_vocab_size = self.encoder.config.vocab_size

        if tokenizer_vocab_size > curr_vocab_size:
            logger.info(
                f"Tokenizer was found to have vocab size of {tokenizer_vocab_size} while the current \
                vocab size of the model is {curr_vocab_size}. The size of the token embeddings will be \
                resized to match {curr_vocab_size} to ensure consistency."
            )
            self.encoder.resize_token_embeddings(tokenizer_vocab_size)


# class RobertaModelForBinaryTokenClassification(
#     RobertaPreTrainedModel, BinaryTokenClassification
# ):
#     def __init__(self, config: ModelConfig):
#         roberta_config = config._to_roberta_config()
#         super().__init__(config=roberta_config)
#         self.num_labels = config.num_labels
#         # prevent pooling to get token-level outputs, not sentence-level
#         self.model = RobertaModel(config=roberta_config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout
#             if config.classifier_dropout is not None
#             else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(in_features=config.hidden_size, out_features=1)

#         # initialise weights and final pre-processing
#         self.post_init()


class XLMRobertaModelForBinaryTokenClassification(
    XLMRobertaPreTrainedModel, BinaryTokenClassificationModel
):
    def __init__(self, config: ModelConfig):
        roberta_config = config._to_roberta_config()
        super().__init__(config=roberta_config)
        self.num_labels = self.config.num_labels

        # prevent pooling to get token-level, not sentence level outputs
        self.encoder = XLMRobertaModel(config=roberta_config, add_pooling_layer=False)
        self._add_special_tokens(tokenizer=self.encoder.config.tokenizer_class)

        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

        # initialise weights and final processing
        self.__post_init__()

    def forward(
        self,
        input_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        source_mask: torch.BoolTensor,
        target_mask: torch.BoolTensor,
    ):
        return super(XLMRobertaModelForBinaryTokenClassification, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            source_mask=source_mask,
            target_mask=target_mask,
        )

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def _add_special_tokens(self, tokenizer):
        tokenizer_vocab_size = len(tokenizer)
        curr_vocab_size = self.encoder.config.vocab_size

        if tokenizer_vocab_size > curr_vocab_size:
            logger.info(
                f"Tokenizer was found to have vocab size of {tokenizer_vocab_size} while the current \
                vocab size of the model is {curr_vocab_size}. The size of the token embeddings will be \
                resized to match {curr_vocab_size} to ensure consistency."
            )
            self.encoder.resize_token_embeddings(tokenizer_vocab_size)


#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#     ) -> SpanTokenAlignerOutput:
#         return super(RobertaModelForBinaryTokenClassification, self).forward(
#             model=self.model,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#         )

#     def get_input_embeddings(self):
#         return self.model.get_input_embeddings()

#     def set_input_embeddings(self, value):
#         self.model.set_input_embeddings(value)


# class XLMRobertaModelForBinaryTokenClassification(
#     XLMRobertaPreTrainedModel, BinaryTokenClassification
# ):
#     def __init__(self, config: ModelConfig):
#         xlm_roberta_config = config._to_xlm_roberta_config()
#         super().__init__(config=xlm_roberta_config)
#         self.num_labels = config.num_labels
#         # prevent pooling to get token-level outputs, not sentence-level
#         self.model = XLMRobertaModel(config=xlm_roberta_config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout
#             if config.classifier_dropout is not None
#             else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(in_features=config.hidden_size, out_features=1)

#         self.post_init()

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         labels: Optional[torch.Tensor] = None,
#     ) -> SpanTokenAlignerOutput:
#         return super(XLMRobertaModelForBinaryTokenClassification, self).forward(
#             model=self.model,
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#         )

#     def get_input_embeddings(self):
#         return self.model.get_input_embeddings()

#     def set_input_embeddings(self, value):
#         self.model.set_input_embeddings(value)


# TODO: find out if it is better to use RobertaModelForTokenClassification
# TODO: ensure correct configuration set up for each model
# -> num_labels, classifier_dropout, hidden_dropout_prob, hidden_size etc.
# TODO: document shape tracking as tensors move through the model
# TODO: how to add symmetrisation to the model to avoid data complexity?
# TODO: what other algorithms other than cosine similarity can we use to identify if embeddings are similar? (more of a bonus, should focus on completing the ner part first)
