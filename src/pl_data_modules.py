from typing import Any, Union, List, Optional

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.dataset import SenseAnnotatedDataset

from transformers import AutoTokenizer


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

    def prepare_data(self, *args, **kwargs):
        print(os.getcwd())
        # os.system("bash download_dataset.sh")

    def setup(self, stage: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.model.tokenizer)
        self.train_dataset = SenseAnnotatedDataset(
            self.conf, name="semcor", tokenizer=self.tokenizer
        )
        # self.valid_dataset = SenseAnnotatedDataset(self.conf, name='semeval2007', tokenizer=self.tokenizer)
        # self.test_dataset = SenseAnnotatedDataset(self.conf, name='semevalALL', tokenizer=self.tokenizer)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            num_workers=self.conf.data.num_workers,
            batch_size=self.conf.data.batch_size,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.valid_dataset,
            num_workers=self.conf.data.num_workers,
            batch_size=self.conf.data.batch_size,
            shuffle=False,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            num_workers=self.conf.data.num_workers,
            batch_size=self.conf.data.batch_size,
            shuffle=False,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        raise NotImplementedError
