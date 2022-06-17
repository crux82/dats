# Data Augmentation based on Task-specific Similarity

## Introduction

This repository contains the code for the paper _Learning to Generate Examples for Semantic Processing Tasks_ published at 
NAACL 2022 - by Danilo Croce (Tor Vergata, University of Rome), Simone Filice (Amazon), Giuseppe Castellucci (Amazon) 
and Roberto Basili (Tor Vergata, University of Rome).

The paper introduces DATS (Data Augmentation based on Task-specific Similarity), a novel data-augmentation technique
for text classification based on pre-trained large language models. The key idea of DATS is to introduce lexical and structural
variability in the generated examples that are still coherent with the target category they're generated from. In order to achieve this result, 
DATS first learns a task-specific similarity function to create a dataset of example pairs that are then used to fine-tune a pre-trained NLG model.
The fine-tuned model is then used to generate novel examples for the target categories.

Experiments in low resource settings show  that augmenting the training material with the  proposed strategy systematically improves the
results on text classification and natural language inference tasks by up to 10% accuracy, outperforming existing DA approaches.

## Requirements

The code is made of different scripts depending on the following packages:

- simpletransformers==0.61.13
- torch==1.8.1
- transformers==4.8.2


## Installation Instructions

To install the code, we suggest creating a python environment and to install the dependencies listed in the 
_requirements.txt_ file, with:

```
pip install -r requirements.txt
```

After installing the dependencies run

```
pip install .
```

from the main directory of the code. 

## How To Run An Experiment

We release the code that we used to run the experiments of the paper. To replicate an experiment
of the paper, you should use the script to run that is located in 
```
cd scripts
run_experiment.sh
```

This script takes in input: 
- 
- the prefix of an output directory;
- the input data directory containing the _train.txt_, _dev.txt_ and _test.txt_ files
- a random seed
- a gpu index

In the paper, we run each experiment 5 times with 5 different random seeds. To replicate the few-shot experiment on the SST dataset
with 10 example per category, you should run the following commands:

```
sh run_experiment.sh sst5_10 ../../data/sst5_10 1 0
sh run_experiment.sh sst5_10 ../../data/sst5_10 2 0
sh run_experiment.sh sst5_10 ../../data/sst5_10 3 0
sh run_experiment.sh sst5_10 ../../data/sst5_10 4 0
sh run_experiment.sh sst5_10 ../../data/sst5_10 5 0
```

These commands can be executed sequentially or in parallel. If executed in parallel, we suggest to use a different GPU index for each command.
After the execution, 5 directories will be created in output: sst5_10_output_seedX, where X will have the value of a specific random seed.

To compute the performances with the average, as reported in the paper, please use the script in dats/utils/compute_experiments_performance.py, with:

```
python ../dats/utils/compute_experiments_performance.py --experiment-dirs sst5_50_output_seed1/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed2/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed3/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed4/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed5/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/
```

This command will produce in output something similar to:

```

```

## Citation

```
@inproceedings{croce-etal-2022-dats,
    title = "Learning to Generate Examples for Semantic Processing Tasks",
    author = "Croce, Danilo  and
      Filice, Simone and
      Castellucci, Giuseppe  and
      Basili, Roberto",
    booktitle = "Proceedings of North American Association for Computational Linguistics Conference",
    month = jul,
    year = "2022"
}
```