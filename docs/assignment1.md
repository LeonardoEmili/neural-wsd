# Assignment 1 - Neural WSD
In this document, we introduce the project topic, provide an overview of the approach, and prepare the Work Breakdown Structure (WBS) with time estimates for individual tasks.

## Project topic
The project aims at developing a Word Sense Disambiguation (WSD) system to automatically disambiguate ambiguous words into their correct senses. The underlying idea is that the meaning of a word is strongly influenced by its surrounding terms (Firth, 1957). Hence, we should carefully select a good model hypothesis to make accurate predictions to tackle polysemous words.

## Related work
Historically, a large interest has been spent on Word Sense Disambiguation (WSD). Current neural architectures typically tackle the task using knowledge-based techniques or supervised approaches. The former leverages structural information available in Lexical Knowledge Bases (LKBs) as an additional source of prior information. Supervised approaches, instead, frame the problem as a classification task where the goal is to learn the association between words and senses. Along this direction, [Bevilaqua et. al, 2020](https://aclanthology.org/2020.acl-main.255.pdf) incorporate relational information from WordNet 3.0. with a single dot product between the logits layer and a pretrained embedding layer. On the other hand, [Conia et. al, 2021](https://aclanthology.org/2021.eacl-main.286) propose a more general multi-labeling approach to WSD that exploits all gold senses produced by human annotators. More successful approaches in WSD leverage dictionary definitions (i.e., glosses) to produce gloss embeddings and then combine them with contextual information. In this context, [Kumar et. al, 2019](https://aclanthology.org/P19-1568/) initialize the weights of the output layer using the representation obtained from glosses and sense embeddings. Finally, in [Barba et. al, 2021](https://aclanthology.org/2021.naacl-main.371/), they frame the problem as a span extraction task to extract the most suitable span given a target word and the concatenations of all of its gloss definitions.

## Our approach
The idea is to develop a transformer-based approach to exploit the contextual information in the sentences. Leveraging powerful language models (e.g., BERT, RoBERTa), we feed our model with the obtained latent representations of the input words. Moreover, we plan to use pre-trained encoder models to avoid training them from scratch and instead rely on fine-tuning for the downstream task. In this way, we can exploit the large training that most of these language models have been exposed to. Furthermore, we plan to use [WordNet](http://wordnetweb.princeton.edu/perl/webwn) as a sense inventory that, over the years, has become the standard in this task. Finally, we do extensive experimentation to see which feature (e.g., word, lemma, POS), strategy (e.g., sense identification, keeping the word encoders frozen), or classification architecture (e.g., MLP, LSTM) is able to provide the best results.

## Datasets
In this project, we plan to use multiple datasets to carry out a comparative analysis and evaluate the real performances of our model on different domains. The listed datasets have become standard in WSD and provide sense-annotated sentences from a given Lexical Knowledge Bases (LKB). Moreover, all of them are available to download from their publisher sources. For the sake of this project, we apply some preprocessing techniques to extract useful information for our task (e.g., sense identifiers, lemmas, POS tags, etc ...).
- **SemCor**: a manually sense-annotated English corpus consisting of 352 texts from the Brown corpus [\[more\]](https://course.ccs.neu.edu/csg224/ssl/semcor/semcor2.0/doc/semcor.htm)
- **SemEval/Senseval**: a series of evaluation datasets for semantic systems [\[more\]](https://en.wikipedia.org/wiki/SemEval)
- **OMSTI**: a large corpus automatically annotated from parallel corpora and manual annotations [\[paper\]](https://aclanthology.org/K15-1037.pdf)

## Technologies
In this section, we provide an overview of the main technologies considered for this project. We use the popular [Pytorch Lightning](https://www.pytorchlightning.ai/) as a deep learning framework, which builds on top of [Pytorch](https://pytorch.org/) and allows for fast development. To promote reproducibility, we log all the experiments using [Weights and Biases](https://wandb.ai/site) and make them accessible on the project board. For the final deliverable, we consider [Streamlit](https://streamlit.io/) to ship our project on the web as a fully-fledged disambiguation system.

## Work Breakdown Structure
In the following, we decompose the project into phases with time estimates for each one. The time frames allocated to each step are approximations and may be susceptible to slight modifications (e.g., longer training times, more experimentation necessary).
- Datasets collection:  (~1 day)
- Data preprocessing:  (~2 days)
- Data exploration: (~2 days)
- Designing a neural network: (~1 week)
- Training and finetuning: (1-2 weeks)
- Setting an evaluation framework (~1-2 days)
- Testing against a baseline (~2 days)
- Building an interactive demo: (1-2 weeks)
- Writing the final report: (~1 week)
- Preparing the presentation: (2-3 days)
