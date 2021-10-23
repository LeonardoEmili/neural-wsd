# Assignment 1 - Neural WSD
In this document, we introduce the project topic, provide an overview of the approach, and prepare the Work Breakdown Structure (WBS) with time estimates for individual tasks.

## Project topic
The project aims at developing a Word Sense Disambiguation (WSD) system to automatically disambiguate ambiguous words into their correct senses. The underlying idea is that the meaning of a word is strongly influenced by its surrounding terms (Firth, 1957). Hence, we should carefully select a good model hypothesis to make accurate predictions for polysemous words.

## Related work
Historically, a large interest has been spent on Word Sense Disambiguation (WSD). Current neural architectures often frame the problem as a classification task where the goal is to learn the association between words and senses. Along this direction, [Conia et. al, 2021](https://aclanthology.org/2021.eacl-main.286) propose a method to exploit all gold senses available for each target word. On the other hand, the model proposed by [Bevilaqua et. al, 2020](https://aclanthology.org/2020.acl-main.255.pdf) leverages a knowledge-based approach to integrate relational information from WordNet 3.0.

## Our approach
The idea is to develop a transformer-based approach to exploit the contextual information in the sentences. Leveraging powerful language models (e.g., BERT), we feed our model with the obtained latent representations of input words. Moreover, we plan to use pre-trained encoder models to avoid training them from scratch and instead perform fine-tuning on the downstream task. In this way, we also exploit the large training that most of these language models have been exposed to. Furthermore, we plan to use WordNet as a sense inventory that, over the years, has become the standard in this task. Finally, we do extensive experimentation to see which feature (e.g., word, lemma, POS), strategy (e.g., sense identification, keeping the word encoders frozen), or classification architecture (e.g., MLP, LSTM) is able to provide the best results.

## Datasets
In this project, we plan to use multiple datasets to carry out a comparative analysis and evaluate the real performances of our model on different domains. The listed datasets have become standard in WSD and provide sense-annotated sentences from a given Lexical Knowledge Bases (LKB). Moreover, all of them are available to download from their publisher sources. For the sake of this project, we apply some preprocessing techniques to extract useful information for our task (e.g., sense identifiers, lemmas, POS tags, etc ...).
- **SemCor**: a manually sense-annotated English corpus consisting of 352 texts from the Brown corpus
- **SemEval/Senseval**: a series of evaluation datasets for semantic systems
- **OMSTI**: a large corpus automatically annotated from parallel corpora and manual annotations

## Work Breakdown Structure
In the following, we decompose the project into phases with time estimates for each one. The time frames allocated to each step are approximations and may be susceptible to slight modifications (e.g., slower training times, more experimentation necessary).
- Datasets collection:  (~1 day)
- Data preprocessing:  (~2 days)
- Data exploration: (~2 days)
- Designing a neural network: (2-3 days)
- Training and finetuning: (1-2 weeks)
- Setting an evaluation framework (~1 day)
- Testing against a baseline (~2 days)
- Building an interactive demo: (1-2 weeks)
- Writing the final report: (~1 week)
- Preparing the presentation: (2-3 days)

## Author
> Leonardo Emili - e12109608@student.tuwien.ac.at