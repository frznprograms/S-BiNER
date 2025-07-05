Hi there, welcome to the 
## 𝔃eta𝖆lign
reposiotory.

### Description
This repository was designed to implement the BinaryAlign methodology as proposed in the paper (link: https://arxiv.org/pdf/2407.12881), with customizations added for improved readability, and fine-tuning to improve model performance on Chinese word alignment (WA) tasks.

This methodology comes with the added twist of simple annotation projection for Named Entity Recognition (NER) tasks. 

### Author(s)
Shane Bharathan (shanevbh@gmail.com)

### Instructions for usage
First, install the `uv` package management software for Python: https://docs.astral.sh/uv/getting-started/installation/.

Once installed, run the terminal command
```
uv init
```
and then 
```
uv sync
``` 
This will intialise a `.venv` folder and install the required dependencies for this project.

If you do not wish to use the `uv` package (though I would highly recommend it as it's really easy to use and wicked fast), you may also simply run 
```
pip install -r requirements.txt
```

To run inference using **ZetaAlign**, run the following terminal command: 
```

```

### Input and Output formats

### Training Methodology
 
### How the Code Works - A step-by-step guide

#### Dataset Generation
If source_sentence = ["Hello", "world"]
word_by_word_examples becomes:
```
[
  [" [WORD_SEP] ", "Hello", " [WORD_SEP] ", "world"],      # highlighting "Hello"
  ["Hello", " [WORD_SEP] ", "world", " [WORD_SEP] "]       # highlighting "world"
]
```
i.e. `word_by_word_examples` becomes a list[list[str]], a square matrix with an added dimension.

Then source_tokens becomes:
```\
[
  [["[WORD_SEP]"], ["Hello"], ["[WORD_SEP]"], ["world"]],
  [["Hello"], ["[WORD_SEP]"], ["world"], ["[WORD_SEP]"]]
]
```
i.e. `source_tokens` becomes a list[list[list[str]]].

The dimensions of the tensors do **NOT** necessarily match the length of the `source_line`, no doubt due to the magic of PyTorch and ✨ Linear Algebra ✨.

The next step is to do byte-pair encoding:

If token_tgt = [["Hello"], ["world", "!"]] i.e 2 words:
Word 0: "Hello" → 1 BPE token
Word 1: "world!" → 2 BPE tokens  
Result: bpe2word_map_tgt = [0, 1, 1, -1]
        (token 0→word 0, tokens 1,2→word 1, -1 for padding/special)


### Citations
I wish to give credit to the original creators of the concept of BinaryAlign, who also created the repository from which much of the code here was inspired by:

@article{latouche2024binaryalign,
  title={BinaryAlign: Word Alignment as Binary Sequence Labeling},
  author={Latouche, Gaetan Lopez and Carbonneau, Marc-Andr{\'e} and Swanson, Ben},
  journal={arXiv preprint arXiv:2407.12881},
  year={2024}
}