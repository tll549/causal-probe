# Causal Relations and Feature Extraction in Contextualized Sentence Representation

Codes for the paper "Causal Relations and Feature Extraction in Contextualized Sentence Representation".

## Binary causality detection

```
python main.py -probe simple -model ALL -use_pytorch -subset_data downsampling
```

(Probe 1)

Without `-use_pytorch`, it will be run by sklearn LogisticRegression. `-subset_data` controls whether performs downsampling or run on the full dataset.

## Causal feature prediction

(Probe 3)

In order to calculate causal features, OANC dataset must be provided. `OANC_GrAF.zip` should be placed at `data/causal_probing/OANC_GrAF.zip`. It can be downloaded from http://www.anc.org/data/oanc/contents/.

```
python main.py -probe feature -use_pytorch
```

## Causal word prediction with BERT

(Probe 2)

```
python main.py -probe mask
```

`-mask cause/effect` controls which one to mask.

---

`-reset_data 0` will skip the preprocessing, if a processed data is available.

`-reencode_data 0` will skip the encoding data for the first and third probe, if a temporary encoded pickle file is available.



All results will be saved in the `logs` folder.

