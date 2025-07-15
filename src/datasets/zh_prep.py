from dataclasses import dataclass
from typing import Union

import hanlp
import hanlp.pretrained
from loguru import logger


@dataclass
class ZHTokenizer:
    model_name: str = "COARSE_ELECTRA_SMALL_ZH"

    @logger.catch(message="Unable to initialise Hanlp tokenizer!", reraise=True)
    def __post_init__(self):
        if hasattr(hanlp.pretrained.tok, self.model_name):
            model_path = getattr(hanlp.pretrained.tok, self.model_name)
            self.tok = hanlp.load(model_path)
        else:
            try:
                self.tok = hanlp.load(self.model_name)
            except Exception:
                available_models = [
                    attr
                    for attr in dir(hanlp.pretrained.tok)
                    if not attr.startswith("_")
                ]
                logger.error(
                    f"Model '{self.model_name}' not found. "
                    f"Available models: {available_models}"
                )
            logger.warning("Tokenizer has not been initialised!")

    def __call__(
        self, lines: Union[list[str], str], return_as_single_strings: bool = True
    ) -> Union[list[list[str]], list[str], str]:
        res = self.tok(lines)
        if return_as_single_strings:
            # handle str
            if isinstance(res[0], str):
                res = " ".join(res)
            # handle list[str]
            else:
                res = list(map(lambda x: " ".join(x), res))

        return res


if __name__ == "__main__":
    tokenizer = ZHTokenizer()

    # Single text tokenization
    line = "晓美焰来到北京立方庭参观自然语义科技公司"
    tokens = tokenizer(line)
    print(f"Single text tokens: {tokens}")
    print("-" * 50)
    # Multiple texts tokenization
    lines = ["晓美焰来到北京立方庭参观自然语义科技公司", "今天天气很好，适合出去玩。"]
    batch_tokens = tokenizer(lines)
    print(f"Batch tokens: {batch_tokens}")
    print("-" * 50)
    # Example with custom model
    # try:
    #     custom_tokenizer = ZHTokenizer("FINE_ELECTRA_SMALL_ZH")
    #     custom_tokens = custom_tokenizer(line)
    #     print(f"Custom model tokens: {custom_tokens}")
    #     print("-" * 50)
    # except ValueError as e:
    #     print(f"Custom model error: {e}")
