# Assignment 2 - Neural WSD
In this document, we provide an overview of the proposed approach, describe the main steps followed in our implementation, define the evaluation framework, and update the work estimates in the Work Breakdown Structure (WBS) with the actual amount of time spent on each task (where applicable).

## Summary
We tackle the Word Sense Disambiguation (WSD) task leveraging the BERT language model to extract contextualized latent representations of the input words. We rely on English WordNet 3.0 as the target sense inventory. Furthermore, we show the importance of using additional prior information in the form of lexicalizations from the WordNet Lexical Knowledge Base (LKB).

## Experimental setup

### Datasets
We use the Sense-tagged Semantic Corpus (SemCor) 3.0 as the training set, SemEval 2007 as the validation set, and the concatenation of individual SemCor/SensEval datasets as the test set.

### Hyperparameters
In the following table, we present a collection of the hyperparameters used in our experiments.
| Hyperparameter | Value |
| --- | --- |
| model | bert-base-cased |
| loss function | cross entropy |
| optimizer | adam |
| activation | swish |
| max epochs | 10 |
| learning_rate | 1e-3 |
| dropout | 0.2 |
| hidden_size | 256 |
| batch_size | 32 |

### Evaluation metrics
As an evaluation framework, we consider the micro F1 and choose the model with the highest score on the validation set.

### Training details
In order to speed up training, we keep the word embeddings frozen. We average BERT WordPiece embeddings to get back to word-level representations. Moreover, we reduce the number of synsets considered for a given pair (lemma, POS) using WordNet lexicalizations. We use [Weights and Biases](https://wandb.ai/site) to track all the experiments on the [project board](https://wandb.ai/leonardoemili/neural-wsd?workspace=user-leonardoemili).

## Experimental results
In this context, we implement the Most Frequent Sense (MFS) baseline to assess the quality of our system. It is a strong system that outperforms unsupervised WSD systems [\[Hauer, et. al\]](https://arxiv.org/pdf/1808.06729.pdf) and typically reaches 60\% of F1 in the classic setup. Hence, we consider good a WSD system that achieves a score of at least 60-65\% of F1. In the table below, we depict the highest scores achieved by the models on the test set.

| Model | F1 (\%) |
| --- | --- |
| MFS | 61.12 |
| BERT | 64.42 |
| MFS + lexicalization | 67.32 |
| BERT + lexicalization | **69.31** |

## Testing
To promote code quality and prevent failures at production time, we implement unit testing procedures. Typical tests check the behavior of the system in the following areas: code linting, data preprocessing, I/O routines, and model training. Furthermore, we enable Continuous integration (CI) and design the testing workflow that runs thanks to GitHub's actions.

## Work Breakdown Structure
In the following, we show the time spent on each phase.
- Datasets collection:  (2 days)
- Data preprocessing:  (5 days)
- Data exploration: (3 days)
- Designing a neural network: (4 days)
- Training and finetuning: (2 weeks)
- Setting an evaluation framework (2 days)
- Testing against a baseline (1 day)
- Building an interactive demo: (1 week)
- Writing the final report: (3 days)
- Preparing the presentation: (3 days)
