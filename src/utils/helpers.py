from loguru import logger
import pandas as pd


def delist_the_list(items: list):
    for i in range(len(items)):
        items[i] = items[i][0]
    return items


def load_data(paths: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    try:
        src_path = paths["src_data"]
        tgt_path = paths["tgt_data"]
        alignments_path = paths["align_data"]

    except KeyError as k:
        logger.error(
            f"Necessary keys for loading data could not be found. Pleae ensure that the keys are 'src_data', 'tgt_data' and 'align_data. \n Error message: {k}"
        )

    src_data = delist_the_list(pd.read_csv(src_path, sep="\t").values.tolist())
    tgt_data = delist_the_list(pd.read_csv(tgt_path, sep="\t").values.tolist())
    align_data = delist_the_list(pd.read_csv(alignments_path, sep="\t").values.tolist())

    return src_data, tgt_data, align_data
