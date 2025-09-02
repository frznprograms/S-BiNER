from torch.utils.data import Dataset


class SBinerNERDataset(Dataset):
    def __init__(self, source_sentences, target_sentences):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        return len(self.target_sentences)

    def __getitem__(self, idx: int):
        return self.target_sentences[idx]
