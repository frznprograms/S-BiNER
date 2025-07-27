<h1 align="center">S-BiNER</h1>
<h3 align="center">Binary Word Alignment for NER and Cross-Lingual Tranfer Tasks</h3>

### Description
This repository was designed to implement the BinaryAlign methodology as proposed in the paper (link: https://arxiv.org/pdf/2407.12881), with customizations added for improved readability, adaptability to different training devices and fine-tuning to improve model performance on multilingual word alignment (WA) and NER tasks, especially on low-resource languages like Chinese. Along the way, I decided to reframe the training objective as truly binary, i.e. a token-to-token alignment problem. This leads to a higher time and space complexity, with the desired effect of better performance. 

The methodology here is intended to come with the added twist of simple heuristics-based annotation projection for Named Entity Recognition (NER) tasks, after the WA model has done all the heavy-lifting.

This project is a work-in-progress, and I welcome constructive feedback into any ways we can improve the way the model is created, trained, and evaluated. Feel free to reach out to me via GitHub.

**Disclaimer**: This framework is experimental and not designed for production-grade tasks. I am NOT selling this product for profit or for any monetary gain at all. In the spirit of collaboration and community-supported improvements, I wish to keep this repo free-to-use and open-source. ***Please include the citations included below if you wish to repurpose this work.*** 

### Author(s)
Shane Vivek

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


🚧 WORK IN PROGRESS 🚧

### Input and Output formats


### License


### Citations
I wish to give credit to the original creators of the concept of BinaryAlign, who also created the repository from which much of the code here was inspired, as well as the creators of the RoBERTa and XLM-R model, which were both utilised in this repo. 

I would also like to thank the good folks at Natural Semantics (Qingdao) Technology Co., Ltd. the main project owner of Hanlp, and Shanghai Linyuan Company for allowing the use of their tokenizer and models for research and teaching purposes. 

**Please include these citations if this work is reproduced or repurposed in any way.**

```
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
```