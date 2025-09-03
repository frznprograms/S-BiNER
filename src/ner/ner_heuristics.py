from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.ner_dataset import SBinerNERDataset
from src.utils.helpers import get_num_workers


@dataclass
class NERHeuristics:
    target_labels_list: list[list[str]] = field(default_factory=list)
    batch_size: int = 4
    debug_mode: bool = False

    @logger.catch(message="Unable to complete NER labelling.", reraise=True)
    def run(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        source_labels: list[list[str]],
        alignments: list[list[tuple[int, int]]],
        default_max_workers: int = 3,
    ):
        # TODO: use multithreading for speed?

        num_workers = get_num_workers(default_max_workers)
        if self.debug_mode:
            logger.info(f"Using {num_workers} cpu cores.")

        # Each dataset item should yield (src_sent, tgt_sent, src_labels, alignment)
        dataset = SBinerNERDataset(
            source_sentences=source_sentences,
            target_sentences=target_sentences,
            source_labels=source_labels,
            alignments=alignments,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: list(zip(*x)),
            num_workers=num_workers,
        )

        pbar = tqdm(
            total=len(target_sentences), desc="Performing annotation projection..."
        )
        projected_ner_labels = []

        for (
            batched_source_sentences,
            batched_target_sentences,
            batched_source_labels,
            batched_alignments,
        ) in dataloader:
            batched_alignment_dicts = self._prepare_alignments_dict(
                alignments=batched_alignments
            )
            batched_entity_spans = self.identify_ner_tags(
                source_labels=batched_source_labels
            )
            batched_target_labels = self.project_labels_batched(
                target_labels=self._prepare_target_labels_list(
                    batched_target_sentences
                ),
                alignments_dict_list=batched_alignment_dicts,
                entity_spans=batched_entity_spans,
            )

            pbar.update(len(batched_target_sentences))
            projected_ner_labels.extend(batched_target_labels)

        return projected_ner_labels

    @logger.catch(
        message="Unable to identify NER tags for batched sentence(s).", reraise=True
    )
    def identify_ner_tags(
        self, source_labels: list[list[str]]
    ) -> list[list[tuple[int, int, str]]]:
        """Groups entities by tags: 'B-LOC', 'I-LOC', 'I-LOC' becomes 'LOC', 'LOC', 'LOC'"""
        entity_spans = []
        for i in range(len(source_labels)):
            entity_spans_single = []
            source_labels_single = source_labels[i]
            i = 0
            while i < len(source_labels_single):
                label = source_labels[i]
                if label.startswith("B-"):  # type: ignore
                    start = i
                    entity_type = label[2:]
                    i += 1
                    # ensure entity type is correct
                    while (
                        i < len(source_labels)
                        and source_labels[i] == f"I-{entity_type}"
                    ):
                        i += 1
                    entity_spans_single.append((start, i, entity_type))
                else:
                    i += 1
            entity_spans.append(entity_spans_single)

        return entity_spans

    @logger.catch(message="Unable to prepare batched target labels.", reraise=True)
    def _prepare_target_labels_list(
        self, target_sentences: list[str]
    ) -> list[list[str]]:
        target_labels = []
        for i in range(len(target_sentences)):
            target_sentence = target_sentences[i]
            target_labels_single = ["O" * len(target_sentence.split())]
            target_labels.append(target_labels_single)

        return target_labels

    @logger.catch(
        message="Unable to prepare batched alignments dictionary.", reraise=True
    )
    def _prepare_alignments_dict(
        self, alignments: list[list[tuple[int, int]]]
    ) -> list[dict[int, list[int]]]:
        alignments_dict = []
        for i in range(len(alignments)):
            alignments_dict_single = defaultdict(list)
            alignments_single = alignments[i]
            for source_idx, target_idx in alignments_single:
                alignments_dict_single[source_idx].append(target_idx)

            alignments_dict.append(alignments_dict_single)

        return alignments_dict

    @logger.catch(message="Unable to check for alignments.", reraise=True)
    def _check_for_alignments(
        self, alignments_dict: dict[int, list[int]], idx: int
    ) -> list[int]:
        return alignments_dict.get(idx, [])

    @logger.catch(message="Unable to execute batched label projection.", reraise=True)
    def project_labels_batched(
        self,
        target_labels: list[list[str]],
        alignments_dict_list: list[dict[int, list[int]]],
        entity_spans: list[list[tuple[int, int, str]]],
    ) -> list[list[str]]:
        # all lengths should be consistent
        assert (
            len(target_labels) == len(alignments_dict_list) == len(entity_spans)
        ), "Please ensure that the lists are all the same length"

        projected_labels = []
        n = len(target_labels)
        for i in range(n):
            target_labels_single, alignments_dict_single, entity_spans_single = (
                target_labels[i],
                alignments_dict_list[i],
                entity_spans[i],
            )
            projected_labels_single = self.project_labels(
                target_labels=target_labels_single,
                alignments_dict=alignments_dict_single,
                entity_spans=entity_spans_single,
            )
            projected_labels.append(projected_labels_single)

        return projected_labels

    @logger.catch(
        message="Unable to project labels from source to target sentence.", reraise=True
    )
    def project_labels(
        self,
        target_labels: list[str],
        alignments_dict: dict[int, list[int]],
        entity_spans: list[tuple[int, int, str]],
    ) -> list[str]:
        if not alignments_dict or not entity_spans:
            logger.error(
                "Please pepare the sentence alignments using \
                    the _prepare_alignments_dict method and the entity spans \
                    using the identify_ner_tags method."
            )
        for start_entity_idx, end_entity_idx, entity_type in entity_spans:
            an_alignment = self._check_for_alignments(
                alignments_dict=alignments_dict, idx=start_entity_idx
            )
            # recursively check for alignment in subsequent words
            while not an_alignment and start_entity_idx <= end_entity_idx:
                start_entity_idx += 1
                an_alignment = self._check_for_alignments(
                    alignments_dict=alignments_dict, idx=start_entity_idx
                )
            start_entity_alignments = an_alignment
            if not start_entity_alignments:
                # if start_idx = end_idx, no alignment -> skip entity
                continue

            an_alignment = self._check_for_alignments(
                alignments_dict=alignments_dict, idx=end_entity_idx
            )
            while not an_alignment and start_entity_idx <= end_entity_idx:
                end_entity_idx += 1
                an_alignment = self._check_for_alignments(
                    alignments_dict=alignments_dict, idx=end_entity_idx
                )
            end_entity_alignments = an_alignment
            if not end_entity_alignments:
                continue

            num_start_alignments = len(start_entity_alignments)
            num_end_alignments = len(end_entity_alignments)

            if num_start_alignments == num_end_alignments == 1:
                # map start, and end, then everything else in between becomes
                # I-<code>
                target_start_entity_idx = start_entity_alignments[0]
                target_end_entity_idx = end_entity_alignments[0]
                target_labels[target_start_entity_idx] = f"B-{entity_type}"
                target_labels[target_start_entity_idx + 1 : target_end_entity_idx] = (
                    f"I-{entity_type}"
                )
                continue  # skip the next part

            # if there is more than one alignment, we will take the alignments
            # that maximise the alignment span -> will be more conservative in projection
            if num_start_alignments > 1:
                min_target_idx = min(start_entity_alignments)

            if num_end_alignments > 1:
                max_target_idx = max(end_entity_alignments)

            target_labels[min_target_idx] = f"B-{entity_type}"  # type: ignore
            target_labels[min_target_idx + 1 : max_target_idx] = f"I-{entity_type}"  # type: ignore

        return target_labels
