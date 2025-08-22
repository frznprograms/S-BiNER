from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class NERHeuristics:
    target_sentence: str
    alignments_dict: dict[int, list[int]] = field(default_factory=dict)
    entity_spans: list[tuple[int, int, str]] = field(default_factory=list)
    target_labels: list[str] = field(default_factory=list)

    @logger.catch(message="Unable to complete NER labelling.", reraise=True)
    def run(self):
        pass

    @logger.catch(message="Unable to identify NER tags for sentence.", reraise=True)
    def identify_ner_tags(self, source_labels: list[str]) -> list[tuple[int, int, str]]:
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
    def prepare_target_labels_list(self):
        self.target_labels = ["O" * len(self.target_sentence.split())]

    @logger.catch(message="Unable to prepare alignments dictionary.", reraise=True)
    def prepare_alignments_dict(self, alignments: list[tuple[int, int]]):
        self.alignments_dict = defaultdict(list)
        for source_idx, target_idx in alignments:
            self.alignments_dict[source_idx].append(target_idx)

    @logger.catch(message="Unable to check for alignments.", reraise=True)
    def check_for_alignments(self, idx: int) -> list[int]:
        return self.alignments_dict.get(idx, [])

    @logger.catch(
        message="Unable to project labels from source to target sentence.", reraise=True
    )
    def project_labels(self):
        if not self.alignments_dict or not self.entity_spans:
            logger.error(
                "Please pepare the sentence alignments using \
                    the prepare_alignments_dict method and the entity spans \
                    using the identify_ner_tags method."
            )
        for start_entity_idx, end_entity_idx, entity_type in self.entity_spans:
            start_entity_alignments = self.check_for_alignments(idx=start_entity_idx)
            end_entity_alignments = self.check_for_alignments(idx=end_entity_idx)
            num_start_alignments = len(start_entity_alignments)
            num_end_alignments = len(end_entity_alignments)

            # handle easiest case: num_start_alignments == num_end_alignments == 1
            if num_start_alignments == num_end_alignments == 1:
                target_start_entity_idx = start_entity_alignments[0]
                target_end_entity_idx = end_entity_alignments[0]
                self.target_labels[target_start_entity_idx] = f"B-{entity_type}"
                self.target_labels[
                    target_start_entity_idx + 1 : target_end_entity_idx
                ] = f"I-{entity_type}"

            # TODO: account for more edge cases
