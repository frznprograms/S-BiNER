from dataclasses import dataclass, field

import hanlp
import hanlp.pretrained
import pandas as pd
from loguru import logger
from typing_extensions import Self

from src.datasets.base_dataset import BaseDataset
from src.utils.decorators import timed_execution


@dataclass
class AwesomeAlignDataset(BaseDataset):
    data_path: str
    context_sep: str = " ||| "  # as prescribed in AwesomeAlign repo
    data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    save: bool = True
    save_path: str = "data/raw_data/data.csv"

    def __post_init__(self):
        # hardcoded oops
        logger.info("Loading Chinese tokenizer...")
        self.source_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        self.data: pd.DataFrame = self.prepare_data(
            path=self.data_path, include_reverse=True
        )
        if self.save:
            self.save_data(save_path=self.save_path)
        # self.target_tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

        logger.success("AwesomeAlignDataset initialised.")

    def __len__(self) -> int:
        return len(self.data)

    @timed_execution
    def prepare_data(self, path: str, include_reverse: bool = True) -> pd.DataFrame:
        logger.info(f"Reading data from {path}...")
        if not path.endswith(".txt"):
            logger.error("TypeError: data file must be of type .txt")

        lines = self.read_data(path=path)

        source_sentences: list[str] = []
        target_sentences: list[str] = []
        for i, line in enumerate(lines):
            if i % 2 == 0:
                source: str = line.strip()
                source_sentences.append(source)
                if i + 1 < len(lines):
                    target: str = lines[i + 1].strip()
                    target_sentences.append(target)

        # batch tokenize with chinese-optimized tokenizer
        source_sentences = self.source_tok(source_sentences)
        source_sentences = list(map(lambda x: " ".join(x), source_sentences))
        temp_df = pd.DataFrame({"source": source_sentences, "target": target_sentences})
        temp_df["final"] = temp_df["source"] + self.context_sep + temp_df["target"]
        if include_reverse:
            temp_df["reverse"] = (
                temp_df["target"] + self.context_sep + temp_df["source"]
            )
            final_data = pd.concat(
                [temp_df["final"], temp_df["reverse"]], axis=0, ignore_index=True
            )
            result_df = pd.DataFrame({"final": final_data})
        else:
            result_df = temp_df[["final"]]

        logger.success(f"Read data from {path}.")

        return result_df

    def read_data(self, path: str) -> list[str]:
        with open(path, "r") as file:
            lines = file.readlines()

        return lines

    def save_data(self, save_path: str):
        try:
            self.data.to_csv(save_path, index=False)
            logger.success(f"Data saved to {save_path}.")
        except Exception as e:
            logger.error(f"Unable to save dataset: {e}")

    def combine_data(self, others: list[Self], override: bool = False) -> pd.DataFrame:
        logger.info(
            "Combining AwesomeAlignDataset(s) now, please ensure you are joining the correct datasets to avoid duplicate data..."
        )
        all_dataframes = [self.data] + [other.data for other in others]
        combined_df = pd.concat(all_dataframes, axis=0)

        if override:
            self.data = combined_df

        return combined_df


if __name__ == "__main__":
    un = AwesomeAlignDataset(
        data_path="data/raw_data/UN.txt",
        save=True,
        save_path="data/raw_data/un.csv",
    )
    hk = AwesomeAlignDataset(
        data_path="data/raw_data/HK.txt",
        save=True,
        save_path="data/raw_data/hk.csv",
    )
    new_df = un.combine_data([hk], override=True)
    un.save_data(save_path="data/cleaned_data/aadf_2.csv")

# see example UN 1200+ for issues
# can we financially justify the use of DeepL to create a parallel corpora, after the model has proven to train/work?
