# Causal Relations and Feature Extraction in Contextualized Sentence Representation

Codes for the paper "Causal Relations and Feature Extraction in Contextualized Sentence Representation".

# Dependencies

This code is written in python. The dependencies are:

- Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
- [Pytorch](http://pytorch.org/)>=1.3.1
- [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.22.0
- [Pandas](https://pandas.pydata.org/)>=0.25.3
- [nltk](https://www.nltk.org/)>=3.4.5
- [transformers](https://huggingface.co/transformers/)>=2.3.0

# Dataset

The [SemEval](http://semeval2.fbk.eu/semeval2.php) #8 dataset is provided at `data/causal_probing/SemEval_2010_8/raw`. It can also be found [here](http://semeval2.fbk.eu/semeval2.php).

# Basic Usage

## Binary causality detection

```
python main.py -probe simple -model ALL -use_pytorch -subset_data downsampling
```

## Causal feature prediction

In order to calculate causal features, OANC dataset must be provided. `OANC_GrAF.zip` should be placed at `data/causal_probing/OANC_GrAF.zip`. It can be downloaded from http://www.anc.org/data/oanc/contents/.

```
python main.py -probe feature -model ALL -use_pytorch
```

## Causal word prediction with BERT

```
python main.py -probe mask
```

`-mask cause/effect` controls which one to mask.

# Other configuration

For all probes,

- `-reset_data 0` will skip the preprocessing, if a processed data is available.

For the first and the second probe,

- `-reencode_data 0` will skip the encoding data for the first and third probe, if a temporary encoded pickle file is available.

- Without `-use_pytorch`, it will be run by sklearn LogisticRegression. 
- `-model ALL` run the probing task for all four models. Can be `['bert', 'gpt2', 'glove', 'conceptnet']`
- `-subset_data` controls whether performs downsampling or run on the full dataset.
- `-cv` controls the number of folds for cross validation, the default is 5.
- `-seed` specify the seed for all random states.



All results will be saved in the `logs` folder.

