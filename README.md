Hi there, welcome to the 
## ZetaAlign
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


### Citations
I wish to give credit to the original creators of the concept of BinaryAlign, who also created the repository from which much of the code here was inspired by, as well as the creators of the RoBERTa and XLM-R model, which were both utilised in this repo.

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