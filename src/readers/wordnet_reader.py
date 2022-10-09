from collections import Counter, defaultdict
from operator import itemgetter
from typing import *

from torchtext.vocab import Vocab
from omegaconf import DictConfig
import hydra

from src.utils.utilities import vocab_lookup_indices, read_json_hydra
from src.readers.raganato_reader import RaganatoReader


class WordNetReader(object):
    """A WordNet reader class that implements the Singleton pattern."""

    _conf = None
    _vocab = None
    _glosses = None
    _lexeme_means = None
    _lemma_means = None
    _sense_means = None

    @classmethod
    def vocabulary(cls, conf: Optional[DictConfig] = None) -> Vocab:
        cls._conf = cls._conf or conf
        if cls._glosses is None:
            cls._glosses = read_json_hydra(path=cls._conf.data.wordnet.glosses)
        cls._vocab = cls._vocab or Vocab(Counter(cls._glosses.keys()), specials=["<pad>", "<unk>"])
        return cls._vocab

    @classmethod
    def sense_means(cls, conf: Optional[DictConfig] = None) -> defaultdict:
        cls._conf = cls._conf or conf
        if cls._sense_means is None:
            cls._sense_means = defaultdict(int, read_json_hydra(path=cls._conf.data.wordnet.sense_means))
        return cls._sense_means

    @classmethod
    def mfs_lexeme_means(cls, conf: Optional[DictConfig] = None) -> defaultdict:
        if cls._lexeme_means is None:
            cls.lexeme_means(conf)
            means = {k: Counter(v).most_common(1)[0][0] for k, v in cls._lexeme_means.items()}
            cls._wn_mfs_lexeme_means = defaultdict(int, means)
        return cls._wn_mfs_lexeme_means

    @classmethod
    def lexeme_means(cls, conf: Optional[DictConfig] = None, vocab: Optional[Vocab] = None) -> defaultdict:
        cls._conf = cls._conf or conf
        cls._vocab = cls._vocab or vocab
        if cls._lexeme_means is None:
            lexeme_means = read_json_hydra(path=cls._conf.data.wordnet.lexeme_means)
            lexeme_means = {k: vocab_lookup_indices(cls._vocab, v) for k, v in lexeme_means.items()}
            cls._lexeme_means = defaultdict(list, lexeme_means)
        return cls._lexeme_means

    @classmethod
    def lemma_means(cls, conf: Optional[DictConfig] = None, vocab: Optional[Vocab] = None) -> defaultdict:
        cls._conf = cls._conf or conf
        cls._vocab = cls._vocab or vocab
        if cls._lemma_means is None:
            lemma_means = read_json_hydra(path=cls._conf.data.wordnet.lemma_means)
            lemma_means = {k: vocab_lookup_indices(cls._vocab, v) for k, v in lemma_means.items()}
            cls._lemma_means = defaultdict(list, lemma_means)
        return cls._lemma_means
