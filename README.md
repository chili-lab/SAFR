# SAFR

This is the offical implementation of [SAFR: Neuron Redistribution for Interpretability (2025)](https://arxiv.org/abs/2501.16374).

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)

## Datasets
The required datasets are automatically downloaded as part of the code execution. Alternatively, you can manually download:

[SST-2 dataset](https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)

[IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

## Usage
You can select the dataset and set the regularization parameters directly via command-line arguments.

For example, to run training on the SST2 dataset with both regularization weights set to 1:

```bash
python train.py --dataset sst2 --importance_lambda 1 --interaction_lambda 1
```

## Citation

### BibTeX Code Block:
To format a BibTeX citation:
```markdown
```bibtex
@article{chang2025safr,
  title={SAFR: Neuron Redistribution for Interpretability},
  author={Chang, Ruidi and Deng, Chunyuan and Chen, Hanjie},
  journal={arXiv preprint arXiv:2501.16374},
  year={2025}
}
```
```
