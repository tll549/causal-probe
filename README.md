# Finding the root cause: Pre-trained encoders' use of world knowledge and sentence understanding for causality sensitivity

Code for the paper "Finding the root cause: Pre-trained encoders' use of world knowledge and sentence understanding for causality sensitivity".

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
python main.py -probe simple -model ALL -use_pytorch
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
- `-model_type base/large` controls whether to use BERT base or large.

For the first and the second probe,

- `-reencode_data 0` will skip the encoding data for the first and third probe, if a temporary encoded pickle file is available.

- Without `-use_pytorch`, it will be run by sklearn `LogisticRegression`. 
- `-model ALL` run the probing task for all four models. Can be `['bert', 'gpt2', 'glove', 'conceptnet']`.
- `-subset_data` controls whether performs downsampling or run on the full dataset. Either "all" or "downsampling".
- `-cv` controls the number of folds for cross validation, the default is 5.
- `-seed` specify the seed for all random states.



All results will be saved in the `logs` folder.

# Other details

The runtime for three tasks above are approximately,

1. 5/20 minutes excluding/including preprocessing and encoding time
2. 50/230 minutes excluding/including preprocessing and encoding time
3. 60/60 minutes excluding/including preprocessing and encoding time

running on aIntel Xeon CPU E5-2678 v3 2.50GHz, for example.



The number of parameters in our logistic regression classifiers depend on the dimension of  the encoder being used. Usually around a thousand. Take BERT base for example, it has 768 dimension so that the corresponding classifier will have around 768*2 parameters with L2 regularization. Four encoders, BERT, GPT-2, GloVe, and ConceptNet have 768, 768, 300, 300 dimensions, respectively.