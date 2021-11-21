from collections import defaultdict
from typing import Optional

from pytorch_lightning.metrics.functional import f1 as f1_score
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch

from src.layers.word_encoder import WordEncoder
from src.utils.torch_utilities import RAdam
from src.models.wsd_model import WSDModel


class BasePLModule(pl.LightningModule):
    def __init__(
        self,
        conf: DictConfig,
        n_classes: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conf = conf
        self.n_classes = n_classes
        self.loss_function = nn.CrossEntropyLoss()
        self.save_hyperparameters({**dict(conf), "n_classes": n_classes})
        self.model = WSDModel(conf, n_classes=n_classes)

    def _evaluate(self, x: dict[str, torch.Tensor], logits_: torch.Tensor, labels: torch.Tensor):
        mask = labels != 0
        logits, labels = logits_[mask], labels[mask]
        loss = F.cross_entropy(logits, labels)

        if "sense_mask" in x:
            logits_.masked_fill_(~x["sense_mask"], float("-inf"))

        annotation = torch.argmax(logits_[mask], dim=-1)
        f1_micro = f1_score(annotation, labels, num_classes=self.n_classes, average="micro")
        return f1_micro, loss

    def _shared_step(self, x: dict[str, torch.Tensor]):
        logits = self.model(x)
        f1_micro, loss = self._evaluate(x, logits, labels=x["senses"])
        return f1_micro, loss

    def training_step(self, x: dict, batch_idx: int) -> dict:
        f1_micro, loss = self._shared_step(x)
        metrics = {"f1_micro": f1_micro, "loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def validation_step(self, x: dict, batch_idx: int) -> dict:
        f1_micro, loss = self._shared_step(x)
        metrics = {"val_f1_micro": f1_micro, "val_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def test_step(self, x: dict, batch_idx: int) -> dict:
        f1_micro, loss = self._shared_step(x)
        metrics = {"test_f1_micro": f1_micro, "test_loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
            loss avg 3.650 - f1 avg 00255
        """

        # return RAdam(self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.model.learning_rate)
        return optimizer
