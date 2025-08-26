from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.ner_dataset import SBinerNERDataset


@dataclass
class NERHeuristics:
    target_sentences: list[str]
    source_labels: list[list[str]]
    # alignments_dict_list: list[dict[int, list[int]]] = field(default_factory=list)
    # entity_spans_list: list[list[tuple[int, int, str]]] = field(default_factory=list)
    target_labels_list: list[list[str]] = field(default_factory=list)
    batch_size: int = 4
    debug_mode: bool = False

    @logger.catch(message="Unable to complete NER labelling.", reraise=True)
    def run(self, default_max_workers: int = 3):
        # TODO: instead of storing all the alignments and entity spans, just return
        # the list of target labels -> more memory efficient and no need dataloader
        # TODO: use multithreading for speed?

        num_workers = get_num_workers(default_max_workers)
        if self.debug_mode:
            logger.info(f"Using {num_workers} cpu cores.")
        # create dataset, then create alignments_dict and entity_spans on the fly
        dataset = SBinerNERDataset(self.target_sentences)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x,  # just return the batch
            num_workers=num_workers,
        )

        pbar = tqdm(
            total=len(self.target_sentences), desc="Performing annotation projection..."
        )
        projected_ner_labels = []
        for batched_source_labels, batched_target_sentences in dataloader:
            batched_alignment_dicts = self._prepare_batched_alignment_dicts()
            batched_entity_spans = self.identify_batched_ner_tags(batched_source_labels)
            # TODO: source labels need to be part of SBinerNERDataset
            batched_target_labels = self.project_labels_batched()

            pbar.update(len(batched_target_labels))

    @logger.catch(message="Unable to identify NER tags for sentence.", reraise=True)
    def identify_ner_tags(self, source_labels: list[str]) -> list[tuple[int, int, str]]:
        """Groups entities by tags: 'B-LOC', 'I-LOC', 'I-LOC' becomes 'LOC', 'LOC', 'LOC'"""
        entity_spans = []
        i = 0
        while i < len(source_labels):
            label = source_labels[i]
            if label.startswith("B-"):
                start = i
                entity_type = label[2:]
                i += 1
                # ensure entity type is correct
                while i < len(source_labels) and source_labels[i] == f"I-{entity_type}":
                    i += 1
                entity_spans.append((start, i, entity_type))
            else:
                i += 1

        return entity_spans

    @logger.catch(message="Unable to prepare target labels.", reraise=True)
    def _prepare_target_labels_list(self, target_sentence: str) -> list[str]:
        target_labels = ["O" * len(target_sentence.split())]
        return target_labels

    @logger.catch(message="Unable to prepare alignments dictionary.", reraise=True)
    def _prepare_alignments_dict(
        self, alignments: list[tuple[int, int]]
    ) -> dict[int, list[int]]:
        alignments_dict = defaultdict(list)
        for source_idx, target_idx in alignments:
            alignments_dict[source_idx].append(target_idx)

        return alignments_dict

    @logger.catch(message="Unable to check for alignments.", reraise=True)
    def _check_for_alignments(
        self, alignments_dict: dict[int, list[int]], idx: int
    ) -> list[int]:
        return alignments_dict.get(idx, [])

    @logger.catch(
        message="Unable to project labels from source to target sentence.", reraise=True
    )
    def project_labels(
        self,
        target_labels: list[str],
        alignments_dict: dict[int, list[int]],
        entity_spans: list[tuple[int, int, str]],
    ):
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
