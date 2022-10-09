import os
import subprocess
from collections import defaultdict
from functools import partial
from typing import *

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from transformers import AutoTokenizer

from src.readers.wordnet_reader import WordNetReader
from src.dataset import SenseAnnotatedDataset
from src.utils.utilities import *


class BasePLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.prepare_data()
        self.setup()

    def prepare_data(self, *args, **kwargs) -> None:
        base_path = hydra.utils.to_absolute_path(".")
        if not os.path.exists(os.path.join(base_path + "/data/", "WSD_Training_Corpora/")):
            subprocess.run(f"python src/scripts/wordnet_extractor.py", shell=True, check=True, cwd=base_path)
            subprocess.run(f"bash src/scripts/get-wsd-data.sh", shell=True, check=True, cwd=base_path)

    def setup(self, stage: Optional[str] = None) -> None:
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        self.train_dataset = SenseAnnotatedDataset.from_cached(
            self.conf,
            tokenizer=self.tokenizer,
            name=self.conf.data.train_ds,
            split="train",
            train_vocab=WordNetReader.vocabulary(self.conf) if self.conf.data.use_synset_vocab else None,
        )
        self.valid_dataset = SenseAnnotatedDataset.from_cached(
            self.conf,
            tokenizer=self.tokenizer,
            name=self.conf.data.val_ds,
            split="validation",
            train_vocab=self.train_dataset.sense_vocabulary,
        )
        self.test_dataset = SenseAnnotatedDataset.from_cached(
            self.conf,
            tokenizer=self.tokenizer,
            name=self.conf.data.test_ds,
            split="test",
            train_vocab=self.train_dataset.sense_vocabulary,
        )

    @property
    def train_features(self) -> Tuple[str]:
        return self.train_dataset.features

    @property
    def sense_vocabulary(self) -> Vocab:
        """Returns the output vocabulary to encode labels (i.e. training or WordNet)."""
        return self.train_dataset.sense_vocabulary

    @property
    def mfs_lexeme_means(self) -> defaultdict:
        if self.conf.data.use_synset_vocab:
            return WordNetReader.mfs_lexeme_means()
        return self.train_dataset.mfs_lexeme_sense_means

    @property
    def lexeme_means(self) -> defaultdict:
        if self.conf.data.use_synset_vocab:
            return WordNetReader.lexeme_means()
        return self.train_dataset.lexeme_senses_means

    @property
    def lemma_means(self) -> defaultdict:
        if self.conf.data.use_synset_vocab:
            return WordNetReader.lemma_means()
        return self.train_dataset.lemma_senses_means

    @property
    def collate_kwargs(self) -> dict[str, any]:
        return {
            "batch_keys": self.train_features,
            "lemma_means": self.lemma_means if self.conf.model.use_lemma_mask else None,
            "lexeme_means": self.lexeme_means if self.conf.model.use_lexeme_mask else None,
            "output_dim": len(self.sense_vocabulary),
        }

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            num_workers=self.conf.data.num_workers,
            batch_size=self.conf.data.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, **self.collate_kwargs),
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.valid_dataset,
            num_workers=self.conf.data.num_workers,
            batch_size=self.conf.data.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, **self.collate_kwargs),
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            num_workers=self.conf.data.num_workers,
            batch_size=self.conf.data.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, **self.collate_kwargs),
        )
