import json
from pathlib import Path
from src.utils.helpers import load_hf_checkpoint
from typing import Callable, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    RobertaPreTrainedModel,
    XLMRobertaPreTrainedModel,
)
from transformers.tokenization_utils import PreTrainedTokenizer
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
        tok_h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, L, H)
        B, L, H = tok_h.shape

        combined_word_ids = torch.cat(
            (source_word_ids, target_word_ids), dim=1
        )  # (B, L)
        pooled_output_ids = self._pool_word_embeddings(
            outputs=tok_h,
            batched_word_ids=combined_word_ids,
            attention_mask=attention_mask.bool(),  # ← bool mask
        )  # (B, W_max, H)

        # counts & maxima
        source_counts_list, target_counts_list, max_source_length, max_target_length = (
            self.compute_word_counts(source_word_ids, target_word_ids)
        )

        # split pooled words into src/tgt and pad to (S_max, T_max)
        pooled_src_emb, pooled_tgt_emb = self.split_and_pad_src_tgt_words(
            pooled_output_ids=pooled_output_ids,
            source_counts_list=source_counts_list,
            target_counts_list=target_counts_list,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )  # (B, S_max, H), (B, T_max, H)

        # cross pairs
        # (B, S_max, T_max, H)
        src_exp = pooled_src_emb.unsqueeze(2).expand(-1, -1, max_target_length, -1)
        # (B, S_max, T_max, H)
        tgt_exp = pooled_tgt_emb.unsqueeze(1).expand(-1, max_source_length, -1, -1)
        pair = torch.cat([src_exp, tgt_exp], dim=-1)  # (B, S_max, T_max, 2H)

        logits = self.classifier(self.dropout(pair)).squeeze(-1)  # (B, S_max, T_max)

        # analyse flow of tensor shapes
        # print(f"Token embedding shapes: {tok_h.shape}")
        # print(f"Combined word ids shape: {combined_word_ids}")
        # print(f"Pooled Output ids shape: {pooled_output_ids.shape}")
        # print(f"Max source sentence length: {max_source_length}")
        # print(f"Max target sentence length: {max_target_length}")
        # print(f"Pooled source embeddings shape: {pooled_src_emb.shape}")
        # print(f"Pooled target embeddings shape: {pooled_tgt_emb.shape}")
        # print(f"Expanded source side embeddings shape: {src_exp.shape}")
        # print(f"Expanded target side embeddings shape: {tgt_exp.shape}")
        # print(f"Final logits shape: {logits.shape}")

        return logits

    @logger.catch(message="Unable to pool embeddings properly", reraise=True)
    def _pool_word_embeddings(
        self,
        outputs: torch.Tensor,  # (B, L, H)
        batched_word_ids: torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor,  # (B, L)  **BOOL**
        agg_fn: Callable = torch.mean,
    ) -> torch.Tensor:
        B, L, H = outputs.shape
        pooled_embeddings = []

        for i in range(B):
            out_i = outputs[i]  # (L, H)
            wid_i = batched_word_ids[i]  # (L,)
            msk_i = attention_mask[i]  # (L,), bool
            word_embs, current_id, buf = [], None, []

            for j in range(L):
                if not msk_i[j]:
                    continue  # skip padded tokens

                wid = int(wid_i[j].item())
                if wid == -1:
                    # skip specials/pad entirely (do not append an embedding)
                    if buf:
                        word_embs.append(agg_fn(torch.stack(buf), dim=0))
                        buf = []
                    current_id = None
                    continue

                if current_id is None or wid != current_id:
                    if buf:
                        word_embs.append(agg_fn(torch.stack(buf), dim=0))
                    buf = [out_i[j]]
                    current_id = wid
                else:
                    buf.append(out_i[j])

            if buf:
                word_embs.append(agg_fn(torch.stack(buf), dim=0))
            if not word_embs:
                # ensure at least one row so pad_sequence works
                word_embs = [out_i.new_zeros(H)]

            pooled_embeddings.append(torch.stack(word_embs))  # (W_i, H)

        return nn.utils.rnn.pad_sequence(
            pooled_embeddings, batch_first=True
        )  # (B, W_max, H)

    def apply_mask(self, sequence_output, mask: torch.BoolTensor):
        B, L, H = sequence_output.size()
        token_lists = []
        for i in range(B):
            indices = mask[i].nonzero(as_tuple=True)[0]
            token_lists.append(sequence_output[i, indices, :])
        return nn.utils.rnn.pad_sequence(token_lists, batch_first=True)

    def compute_word_counts(
        self,
        source_word_ids: torch.Tensor,
        target_word_ids: torch.Tensor,
    ) -> tuple[list[int], list[int], int, int]:
        B = source_word_ids.shape[0]
        source_counts_list = [
            get_unique_words_from_mapping(source_word_ids[i]) for i in range(B)
        ]
        target_counts_list = [
            get_unique_words_from_mapping(target_word_ids[i]) for i in range(B)
        ]
        max_source_length = max(source_counts_list) if source_counts_list else 0
        max_target_length = max(target_counts_list) if target_counts_list else 0

        return (
            source_counts_list,
            target_counts_list,
            max_source_length,
            max_target_length,
        )

    def split_and_pad_src_tgt_words(
        self,
        pooled_output_ids: torch.Tensor,  # (B, W_max, H)
        source_counts_list: list[int],
        target_counts_list: list[int],
        max_source_length: int,
        max_target_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H = pooled_output_ids.shape
        src_list, tgt_list = [], []

        for i in range(B):
            S_i = source_counts_list[i]
            T_i = target_counts_list[i]

            # pooled_output_ids[i] is [src_words || tgt_words] because we pooled after
            # concatenating source_word_ids then target_word_ids
            if S_i > 0:
                src_i = pooled_output_ids[i, :S_i]  # (S_i, H)
            else:
                src_i = pooled_output_ids.new_zeros((1, H))  # keep a row for padding

            if T_i > 0:
                tgt_i = pooled_output_ids[i, S_i : S_i + T_i]  # (T_i, H)
            else:
                tgt_i = pooled_output_ids.new_zeros((1, H))

            src_list.append(src_i)
            tgt_list.append(tgt_i)

        src_word_embs = nn.utils.rnn.pad_sequence(
            src_list, batch_first=True
        )  # (B, S_max′, H)
        tgt_word_embs = nn.utils.rnn.pad_sequence(
            tgt_list, batch_first=True
        )  # (B, T_max′, H)

        # If S_max′/T_max′ differ slightly from provided maxima (due to zeros), pad/truncate:
        if src_word_embs.size(1) < max_source_length:
            pad = src_word_embs.new_zeros(
                (B, max_source_length - src_word_embs.size(1), H)
            )
            src_word_embs = torch.cat([src_word_embs, pad], dim=1)
        else:
            src_word_embs = src_word_embs[:, :max_source_length]

        if tgt_word_embs.size(1) < max_target_length:
            pad = tgt_word_embs.new_zeros(
                (B, max_target_length - tgt_word_embs.size(1), H)
            )
            tgt_word_embs = torch.cat([tgt_word_embs, pad], dim=1)
        else:
            tgt_word_embs = tgt_word_embs[:, :max_target_length]

        return src_word_embs, tgt_word_embs

    @classmethod
    def from_pretrained(
        cls, load_dir: str, map_location: str = "cpu", strict: bool = True
    ):
        load_path = Path(load_dir)

        # 1) Read your custom config (prefer binary_config.json; fall back to config.json for old runs)
        cfg_path = load_path / "binary_config.json"
        if not cfg_path.exists():
            cfg_path = load_path / "custom_config.json"
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # 2) Rebuild ModelConfig
        # Expect either cfg["init_args"]["config"] or cfg["model_config"] to be a dict of ModelConfig fields
        model_cfg_dict = (
            (cfg.get("init_args") or {}).get("config") or cfg.get("model_config") or {}
        )
        model_config = ModelConfig(**model_cfg_dict)

        # 3) Rebuild the encoder
        # Preferred: if the checkpoint contains an /encoder subfolder (saved via encoder.save_pretrained),
        # load from there. Otherwise, fall back to a recorded name_or_path (hub id or local path).
        from transformers import AutoModel

        encoder_subdir = load_path / "encoder"
        name_or_path = (
            cfg.get("backbone_name_or_path")
            or cfg.get("base_model_name_or_path")
            or None
        )

        if encoder_subdir.exists():
            encoder = AutoModel.from_pretrained(str(encoder_subdir), torch_dtype="auto")
        elif name_or_path is not None:
            encoder = AutoModel.from_pretrained(name_or_path, torch_dtype="auto")
        else:
            # Last resort: try root (only works if you placed encoder files in the root, which I don't recommend)
            try:
                encoder = AutoModel.from_pretrained(str(load_path), torch_dtype="auto")
            except Exception as e:
                raise RuntimeError(
                    "Could not locate encoder. Expecting 'encoder/' subfolder or a "
                    "'backbone_name_or_path' in the config."
                ) from e

        # 4) Build the full model and load the weights
        model = cls(encoder=encoder, config=model_config)

        state_dict = load_hf_checkpoint(str(load_path), map_location=map_location)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            logger.warning(
                f"load_state_dict: missing={missing}, unexpected={unexpected}"
            )

        return model


def get_unique_words_from_mapping(word_ids_1d: torch.Tensor) -> int:
    # since we know that special and padding tokens are always encoded as -1,
    # we can just ignore these when we count unique words
    valid = word_ids_1d[word_ids_1d >= 0]
    return int(valid.max().item() + 1) if valid.numel() else 0


def create_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(batch):
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,  # type: ignore
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )

        src_counts = [
            get_unique_words_from_mapping(b["source_token_to_word_mapping"])
            for b in batch
        ]
        tgt_counts = [
            get_unique_words_from_mapping(b["target_token_to_word_mapping"])
            for b in batch
        ]
        max_S = max(src_counts) if src_counts else 0
        max_T = max(tgt_counts) if tgt_counts else 0

        # pad labels + mask
        padded_labels, padded_masks = [], []
        for b in batch:
            L = b["label_matrix"]  # (S_i, T_i)
            S_i, T_i = L.shape
            P = torch.zeros((max_S, max_T), dtype=L.dtype)
            M = torch.zeros((max_S, max_T), dtype=torch.bool)
            P[:S_i, :T_i] = L
            M[:S_i, :T_i] = True
            padded_labels.append(P)
            padded_masks.append(M)

        labels = torch.stack(padded_labels)  # (B, max_S, max_T)
        label_mask = torch.stack(padded_masks)  # (B, max_S, max_T)

        source_word_ids = pad_sequence(
            [b["source_token_to_word_mapping"] for b in batch],
            batch_first=True,
            padding_value=-1,
        )
        target_word_ids = pad_sequence(
            [b["target_token_to_word_mapping"] for b in batch],
            batch_first=True,
            padding_value=-1,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_mask": label_mask,
            "source_word_ids": source_word_ids,
            "target_word_ids": target_word_ids,
        }

    return collate_fn
