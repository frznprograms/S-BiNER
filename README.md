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


### Citations
I wish to give credit to the original creators of the concept of BinaryAlign, who also created the repository from which much of the code here was inspired by:

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