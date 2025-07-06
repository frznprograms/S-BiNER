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