Hi there, welcome to
## S-BiNER ##

### Description
This repository was designed to implement the BinaryAlign methodology as proposed in the paper (link: https://arxiv.org/pdf/2407.12881), with customizations added for improved readability, adaptability to different training devices and fine-tuning to improve model performance on multilingual word alignment (WA) and NER tasks, especially on Chinese.

**Disclaimer**: This model is experimental and not designed for production-grade tasks.

This methodology comes with the added twist of simple annotation projection for Named Entity Recognition (NER) tasks. 

### Author(s)
Shane Bharathan (shanevbh@gmail.com/shane_vbharathan@aiip.sg)

### Instructions for usage
First, install the `uv` package management software for Python: https://docs.astral.sh/uv/getting-started/installation/.

Once installed, run the terminal command
```bash
uv init
```
and then 
```bash
uv sync
``` 
This will intialise a `.venv` folder and install the required dependencies for this project.

If you do not wish to use the `uv` package (though I would highly recommend it), you may also simply run 
```bash
pip install -r requirements.txt
```
To configure accelerate configurations, run 
```bash
accelerate config
```
and follow the prompts in the command line.

To run inference using **S-BiNER**, run the following terminal command: 
```

```

🚧 WORK IN PROGRESS 🚧

### Input and Output formats


### License


### Citations
I wish to give credit to the original creators of the concept of BinaryAlign, who also created the repository from which much of the code here was inspired, as well as the creators of the RoBERTa and XLM-R model, which were both utilised in this repo. 

I would also like to thank the good folks at Natural Semantics (Qingdao) Technology Co., Ltd. the main project owner of Hanlp, and Shanghai Linyuan Company for allowing the use of their tokenizer and models for research and teaching purposes. 

@article{latouche2024binaryalign,
  title={BinaryAlign: Word Alignment as Binary Sequence Labeling},
  author={Latouche, Gaetan Lopez and Carbonneau, Marc-Andr{\'e} and Swanson, Ben},
  journal={arXiv preprint arXiv:2407.12881},
  year={2024}
}

@inproceedings{dou2021word,
  title={Word Alignment by Fine-tuning Embeddings on Parallel Corpora},
  author={Dou, Zi-Yi and Neubig, Graham},
  booktitle={Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2021}
}

@article{DBLP:journals/corr/abs-1911-02116,
  author    = {Alexis Conneau and
               Kartikay Khandelwal and
               Naman Goyal and
               Vishrav Chaudhary and
               Guillaume Wenzek and
               Francisco Guzm{\'{a}}n and
               Edouard Grave and
               Myle Ott and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  journal   = {CoRR},
  volume    = {abs/1911.02116},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.02116},
  eprinttype = {arXiv},
  eprint    = {1911.02116},
  timestamp = {Mon, 11 Nov 2019 18:38:09 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1911-02116.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@article{DBLP:journals/corr/abs-1907-11692,
  author    = {Yinhan Liu and
               Myle Ott and
               Naman Goyal and
               Jingfei Du and
               Mandar Joshi and
               Danqi Chen and
               Omer Levy and
               Mike Lewis and
               Luke Zettlemoyer and
               Veselin Stoyanov},
  title     = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
  journal   = {CoRR},
  volume    = {abs/1907.11692},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11692},
  archivePrefix = {arXiv},
  eprint    = {1907.11692},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-11692.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}