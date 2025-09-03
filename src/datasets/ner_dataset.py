from torch.utils.data import Dataset


class SBinerNERDataset(Dataset):
    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        source_labels: list[list[str]],
        alignments: list[list[tuple[int, int]]],
    ):
        assert (
            len(self.source_sentences)
            == len(self.target_sentences)
            == len(source_labels)
            == len(alignments)
        ), "Please ensure the list are the same length."
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_labels = source_labels
        self.alignments = alignments

    def __len__(self):
        return len(self.target_sentences)

    def __getitem__(self, i: int) -> tuple[str, str, list[str], list[tuple[int, int]]]:
        return (
            self.source_sentences[i],
            self.target_sentences[i],
            self.source_labels[i],
            self.alignments[i],
        )
