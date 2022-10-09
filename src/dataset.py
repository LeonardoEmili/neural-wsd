from collections import Counter, defaultdict
from operator import itemgetter
from typing import *
import logging
import json
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from tqdm import tqdm
from transformers import AutoTokenizer

from src.readers.wordnet_reader import WordNetReader
from src.readers.raganato_reader import *
from src.utils.utilities import *

# A logger for this file
logger = logging.getLogger(__name__)


class SenseAnnotatedDataset(Dataset):
    def __init__(
        self,
        conf: DictConfig,
        tokenizer: Optional[AutoTokenizer] = None,
        senses_vocab: Optional[Vocab] = None,
        name: str = "semcor",
        cached: Optional[Any] = None,
        split: str = "train",
    ) -> None:
        """
        Args:
            conf: the configuration dictionary
            tokenizer: the tokenizer used prepare the data
            name: the name of dataset
            cached: optional, initialize a dataset from pre-computed data
        """
        self.conf: DictConfig = conf
        self.name = name
        self.split = split
        self.tokenizer: AutoTokenizer = tokenizer

        self.load_dataset(cached)
        self.tokenize()

        self.sense_vocabulary = senses_vocab
        self.create_senses_means()

    @property
    def sense_vocabulary(self) -> Vocab:
        return self._sense_vocabulary

    @sense_vocabulary.setter
    def sense_vocabulary(self, vocabulary: Vocab):
        self._sense_vocabulary = vocabulary or self.compute_sense_vocabulary()

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int):
        return self.preprocessed_data[idx]

    def load_dataset(self, cached: Optional[Any] = None) -> None:
        kwargs = {"conf": self.conf, "cached": cached, "split": self.split, "merge_with_semcor": "semcor" in self.name}
        if self.name == "semcor":
            self.data = SemCorReader.read(**kwargs, **self.conf.data.corpora[self.name])
        elif "omsti" in self.name:
            self.data = OMSTIReader.read(**kwargs, **self.conf.data.corpora[self.name])
        elif "semeval" in self.name or "senseval" in self.name:
            self.data = SemEvalReader.read(**kwargs, **self.conf.data.corpora["semeval_all"], filter_ds=self.name)
        else:
            raise ValueError(f"{self.name} Dataset not supported (e.g. try with semcor, semeval2007, ...)")

    @property
    def features(self) -> List[str]:
        return self.preprocessed_data[0].keys()

    def compute_sense_vocabulary(self) -> Vocab:
        counter = Counter([instance_sense for sample in self.data for instance_sense in sample["instance_senses"]])
        return Vocab(counter, specials=["<pad>", "<unk>"])

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return vocab_lookup_indices(vocabulary=self.sense_vocabulary, tokens=tokens)

    def tokenize(self) -> None:
        """Preprocesses the entire dataset."""
        self.preprocessed_data = []
        for sample in tqdm(self.data, desc=f"Tokenizing {self.name}", total=len(self.data)):
            sentence = sample["sentence"]
            lemmas, pos_tags = [], []
            if len(sample["instance_lexemes"]) > 0:
                lemmas, pos_tags = zip(*[lexeme.split("#") for lexeme in sample["instance_lexemes"]])
            sentence_tokenized, indexes = sentence_tokenizer(sentence, self.tokenizer)
            self.preprocessed_data.append(
                {
                    **sentence_tokenized,
                    "lengths": len(sentence),
                    "word_pieces_indexes": indexes,
                    "sentence_id": sample["sentence_id"],
                    "senses_indices": sample["instance_indices"],
                    "lexemes": sample["instance_lexemes"],
                    "lemmas": lemmas,
                }
            )
        assert len(self.data) == len(self.preprocessed_data), "Size mismatch in dataset.tokenize"

    def preprocess_labels(self) -> None:
        """Apply vectorized labels using the correct output vocabulary."""
        msg = f"Creating labels for {self.name}"
        # Show progress bar only when debugging (it's usually really quick)
        for i, sample in tqdm(enumerate(self.data), desc=msg, total=len(self), disable=not self.conf.debug):
            instance_senses = (
                [WordNetReader.sense_means(self.conf)[sense] for sense in sample["instance_senses"]]
                if self.conf.data.use_synset_vocab
                else sample["instance_senses"]
            )
            self.preprocessed_data[i]["senses"] = self.lookup_indices(instance_senses)

    @staticmethod
    def load_data(path: str) -> Dict:
        with open(path, "r") as reader:
            return json.load(reader)

    @staticmethod
    def from_cached(
        conf: DictConfig,
        tokenizer: AutoTokenizer,
        name: str = "semcor",
        split: str = "train",
        train_vocab: Optional[Vocab] = None,
    ):
        """Fetches dataset and vocabulary from file, if not available creates them."""
        base_path = os.path.join(hydra.utils.to_absolute_path("."), conf.data.preprocessed_dir, name)
        ds_path = os.path.join(base_path, "dataset.pth")
        vocab_path = os.path.join(base_path, "vocab.pth")

        # Loads pre-tokenized dataset and senses vocab
        vocab = torch.load(vocab_path) if os.path.exists(vocab_path) and not conf.data.force_preprocessing else None
        dataset = (
            torch.load(ds_path)
            if os.path.exists(ds_path) and not conf.data.force_preprocessing
            else SenseAnnotatedDataset(conf, name=name, tokenizer=tokenizer, split=split, senses_vocab=vocab)
        )
        if vocab is None and (os.path.exists(ds_path) and not conf.data.force_preprocessing):
            # The dataset (hence its vocabulary) is retrieved from file, but the vocab is not
            logger.warning(f"Cannot load vocabulary from {vocab_path}, computing a new one for {split} split.")

        # Persists objects to files
        if conf.data.dump_preprocessed:
            os.makedirs(base_path, exist_ok=True)
            torch.save(dataset, ds_path)
            torch.save(dataset.sense_vocabulary, vocab_path)

        # We can only use training vocabulary at inference time
        dataset.sense_vocabulary = train_vocab

        # Vectorize labels using the given training sense vocabulary
        dataset.preprocess_labels()

        return dataset

    @property
    def lemma_senses_means(self) -> defaultdict:
        return defaultdict(list, self._lemma_senses_means)

    @property
    def lexeme_senses_means(self) -> defaultdict:
        return defaultdict(int, self._lexeme_senses_means)

    @property
    def mfs_lexeme_sense_means(self) -> defaultdict:
        means = {k: Counter(v).most_common(1)[0][0] for k, v in self._lexeme_senses_means.items()}
        return defaultdict(int, means)

    def create_senses_means(self, keys: tuple[str] = ("instance_lexemes", "instance_senses")) -> None:
        """Computes the mapping lexeme->sense."""
        self._lexeme_senses_means = dict()
        for sample in self.data:
            for lexeme, sense in zip(*itemgetter(*keys)(sample)):
                if lexeme not in self._lexeme_senses_means:
                    self._lexeme_senses_means[lexeme] = [self.sense_vocabulary[sense]]
                else:
                    self._lexeme_senses_means[lexeme].append(self.sense_vocabulary[sense])

        self._lemma_senses_means = dict()
        for lexeme, senses in self._lexeme_senses_means.items():
            lemma, pos_tag = lexeme.split("#")
            if lemma not in self._lemma_senses_means:
                self._lemma_senses_means[lemma] = set()
            self._lemma_senses_means[lemma].update(senses)

        # map lemmas to candidate senses (i.e. useful for indexing tensors)
        self._lemma_senses_means = {k: list(v) for k, v in self._lemma_senses_means.items()}
