# Data Augmentation based on Task-specific Similarity

This repository contains the code for the paper _Learning to Generate Examples for Semantic Processing Tasks_ published at 
NAACL 2022 - by Danilo Croce (Tor Vergata, University of Rome), Simone Filice (Amazon), Giuseppe Castellucci (Amazon) 
and Roberto Basili (Tor Vergata, University of Rome). The paper introduces DATS (Data Augmentation based on Task-specific Similarity), a novel data-augmentation technique for text classification based on pre-trained large language models. 

## Introduction

When using recent pre-trained language models, such as BERT (Devlin et al., 2018) or RoBERTa (Liu et al., 2019), the effectiveness
of traditional DA methods (e.g., back-translation or EDA) is extremely limited, and sometimes they can even hurt the results (Longpre et al.,2020). A possible explanation for this inefficacy is that these DA techniques introduce lexical and structural variability that accurate language models directly induce during pre-training. The usefulness of synthetic examples is strictly related to their diversity from the original training data. At the same time, diverging too much from the initial data might increase the risk of introducing noisy annotations, i.e., synthetic data not reflecting the class of the original examples they were generated from. 

To directly tackle the trade-off between diversity and label consistency, we propose DATS (Data Augmentation based on Task-specific Similarity), a novel data-augmentation technique for text classification based on Natural Language Generation (NLG) models.

## Method Overview

To augment the training material for a given NLU task _t_, we propose to fine-tune a NLG model _M_ (e.g., BART) so that it learns to generate synthetic training examples for the task _t_. Specifically, our synthetic example generator has the form _M(c, s<sub>i</sub>)=s<sub>o</sub>_: the model prompt is a class _c_ and an example _s<sub>i</sub>_ of that class; the model output is a new example _s<sub>o</sub>_ which is expected to belong to class _c_ and "inspired" by _s<sub>i</sub>_.
Starting from a reduced set of annotated examples, we first learn a task-oriented similarity function that we use to automatically create pairs of similar examples. Then, we use these pairs to train the model _M_ to generate examples similar to the input one.

The overall data augmentation procedure is exemplified in the figure below:
![DATS schema](https://github.com/crux82/dats/blob/main/training_schema_DATS.png)

1. We split our data into three folds and use the first fold to train a classifier (e.g., BERT). 
2. We convert the examples from the second fold into vectors by running the fine-tuned classifier and taking the output \[CLS\] embedding. We expect these vectors to reflect task-oriented information. 
3. For each class, we compute all pairwise cosine similarities of these vectors and create pairs of similar examples.
4. We train a NLG model (e.g., BART) on the resulting input/output pairs.
5. We use the examples from the third fold to prompt the NLG model and generate new synthetic examples.

We can vary the folds and iterate this procedure to generate more data.

Experiments in low resource settings show  that augmenting the training material with the  proposed strategy systematically improves the
results on text classification and natural language inference tasks by up to 10% accuracy, outperforming existing DA approaches.

## Requirements

The code requires Python3 and the following packages:

- simpletransformers==0.63.7
- torch==1.8.1
- transformers==4.8.2


## Installation Instructions

To install the code, we suggest creating a python environment and to install the dependencies listed in the 
_requirements.txt_ file, with:

```
pip install -r requirements.txt
```

After installing the dependencies run the following command in the root directory of the project:

```
pip install .
```

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
- the input data directory containing the _train.txt_, _dev.txt_ and _test.txt_ files (we're providing the split of data we used for SST with 50 examples)
- a random seed
- a gpu index

In the paper, we run each experiment 5 times with 5 different random seeds. To replicate the few-shot experiment on the SST dataset
with 10 example per category, you should run the following commands:

```
sh run_experiment.sh sst5_50 ../data/sst5_50 1 0
sh run_experiment.sh sst5_50 ../data/sst5_50 2 0
sh run_experiment.sh sst5_50 ../data/sst5_50 3 0
sh run_experiment.sh sst5_50 ../data/sst5_50 4 0
sh run_experiment.sh sst5_50 ../data/sst5_50 5 0
```

These commands can be executed sequentially or in parallel. If executed in parallel, we suggest to use a different GPU index for each command.
After the execution, 5 directories will be created in output: sst5_10_output_seedX, where X will have the value of a specific random seed.

To compute the average performance, as reported in the paper, please use the script in dats/utils/compute_experiments_performance.py. For example:

```
python ../dats/utils/compute_experiments_performance.py --experiment-dirs sst5_50_output_seed1/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed2/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed3/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed4/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/ sst5_50_output_seed5/bert-base-uncased_10epbert_5e-5lr_0.00simbert_5e-5bartlr/
```

This command will produce in output:

```
Model	Validation Avg Acc	Validation Std	Test Avg Acc	Test Std
BERT	0.3868	0.009703607576566572	0.4006	0.011723480711802275
DATS	0.43220000000000003	0.006554387843269582	0.4328	0.01351147660324364
Ttest_indResult(statistic=-3.6000694437746605, pvalue=0.006981594100870437)
```

The output contains the validation average accuracy and standard deviation, the test average accuracy and standard deviation
for the baseline model (BERT) and DATS.
Moreover, the code will output also the p-value for the statistical significance test.

## Citation

```
@inproceedings{croce-etal-2022-learning,
    title = "Learning to Generate Examples for Semantic Processing Tasks",
    author = "Croce, Danilo  and
      Filice, Simone  and
      Castellucci, Giuseppe  and
      Basili, Roberto",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.340",
    pages = "4587--4601",
    abstract = "Even if recent Transformer-based architectures, such as BERT, achieved impressive results in semantic processing tasks, their fine-tuning stage still requires large scale training resources. Usually, Data Augmentation (DA) techniques can help to deal with low resource settings. In Text Classification tasks, the objective of DA is the generation of well-formed sentences that i) represent the desired task category and ii) are novel with respect to existing sentences. In this paper, we propose a neural approach to automatically learn to generate new examples using a pre-trained sequence-to-sequence model. We first learn a task-oriented similarity function that we use to pair similar examples. Then, we use these example pairs to train a model to generate examples. Experiments in low resource settings show that augmenting the training material with the proposed strategy systematically improves the results on text classification and natural language inference tasks by up to 10{\%} accuracy, outperforming existing DA approaches.",
}
```
