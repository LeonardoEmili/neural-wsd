import sys

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from src.pl_data_modules import BasePLDataModule
from src.pl_modules import BasePLModule
from src.models.mfs import MFS
from src.utils.utilities import *


def test(conf: omegaconf.DictConfig) -> None:
    if conf.debug:
        print("Running in DEBUG mode.", file=sys.stderr)

    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = BasePLDataModule(conf)

    # main module declaration
    output_classes = len(pl_data_module.sense_vocabulary)
    checkpoint_path = get_checkpoint_path(conf)
    pl_baseline = MFS(n_classes=output_classes, mfs_means=pl_data_module.mfs_lexeme_means)
    pl_module = BasePLModule.load_from_checkpoint(checkpoint_path, conf=conf, n_classes=output_classes)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, gpus=gpus(conf))

    # module test
    trainer.test(pl_baseline, datamodule=pl_data_module)
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == "__main__":
    main()
