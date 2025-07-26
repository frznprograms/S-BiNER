import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.configs.model_config import ModelConfig
from src.models.binary_token_classification import BinaryTokenClassificationModel


class RobertaModelForBinaryTokenClassification(BinaryTokenClassificationModel):
    def __init__(self, config: ModelConfig):
        roberta_config = AutoConfig.from_pretrained(config.model_name_or_path)
        encoder = AutoModel.from_config(roberta_config)
        super().__init__(encoder, config)
        self.num_labels = self.config.num_labels

        # prevent pooling to get token-level, not sentence level outputs
        # self.encoder = RobertaModel(config=roberta_config, add_pooling_layer=False)
        self.encoder = encoder

        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

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


class XLMRobertaModelForBinaryTokenClassification(BinaryTokenClassificationModel):
    def __init__(self, config: ModelConfig):
        xlm_roberta_config = config._to_xlm_roberta_config()
        encoder = AutoModel.from_config(xlm_roberta_config)
        super().__init__(encoder, config)
        self.num_labels = self.config.num_labels

        # prevent pooling to get token-level, not sentence level outputs
        # self.encoder = XLMRobertaModel(
        #     config=xlm_roberta_config, add_pooling_layer=False
        # )
        self.encoder = encoder
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)

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
