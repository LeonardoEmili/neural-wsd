from collections import defaultdict

from pytorch_lightning.metrics.functional import f1 as f1_score
from torchtext.vocab import Vocab
import pytorch_lightning as pl
import torch

from src.dataset import SenseAnnotatedDataset


class MFS(pl.LightningModule):
    """Implementation of the simple Most Frequent Sense (MFS) baseline"""

    def __init__(self, n_classes: int, mfs_means: defaultdict) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.mfs_means = mfs_means

    def forward(self, x: dict) -> torch.Tensor:
        return torch.tensor([self.mfs_means[l] for lexemes in x["lexemes"] for l in lexemes], device=self.device)

    def test_step(self, x: dict, batch_idx: int) -> dict:
        annotation = self.forward(x)
        labels = x["senses"][x["senses"] != 0]
        f1_micro = f1_score(annotation, labels, num_classes=self.n_classes, average="micro")
        return {"test_f1_micro": f1_micro}
