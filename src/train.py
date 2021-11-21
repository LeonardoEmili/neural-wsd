import sys

from transformers import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import omegaconf
import hydra

from src.pl_data_modules import BasePLDataModule
from src.pl_modules import BasePLModule
from src.utils.utilities import *


def train(conf: omegaconf.DictConfig) -> None:
    if conf.debug:
        print("Running in DEBUG mode.", file=sys.stderr)

    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = BasePLDataModule(conf)

    # main module declaration
    output_classes = len(pl_data_module.sense_vocabulary)
    pl_module = BasePLModule(conf, n_classes=output_classes)

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)

    # logger
    logger: WandbLogger = None
    if conf.logging.log and not conf.debug:
        wandb_login()
        logger: WandbLogger = hydra.utils.instantiate(conf.logging.wandb_logger)
        hydra.utils.log.info(f"W&B is now watching <{conf.logging.watch.log}>!")
        logger.watch(pl_module, log=conf.logging.watch.log, log_freq=conf.logging.watch.log_freq)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, gpus=gpus(conf), logger=logger
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    # store best model path
    if conf.train.model_checkpoint_callback is not None:
        update_latest_checkpoint_path(model_path=model_checkpoint_callback.best_model_path)

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)

    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
