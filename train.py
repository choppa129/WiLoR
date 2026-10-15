from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from yacs.config import CfgNode
from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule
from wilor.models import WiLoR
from wilor.utils.pylogger import get_pylogger
from wilor.utils.misc import task_wrapper, log_hyperparameters

# HACK reset the signal handling so the lightning is free to set it
# Based on https://github.com/facebookincubator/submitit/issues/1709#issuecomment-1246758283
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    if cfg.get('seed') is not None:
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.TRAIN.get('PAPER_MODE', False):
        if cfg.MODEL.get('PAPER_LITERAL', False) and not Path(cfg.MANO.MEAN_PARAMS).is_file():
            raise FileNotFoundError(
                f'Configured MANO initialization file is missing: {cfg.MANO.MEAN_PARAMS}. '
                'Restore that file before training.'
            )
        # The paper does not specify the sampling policy / epoch length.
        # Do not silently call the legacy 100,000-sample worker stream an epoch.
        epoch_samples = cfg.DATASETS.get('TRAIN_SAMPLES_PER_EPOCH_PER_WORKER')
        if not cfg.DATASETS.get('FINITE_FREIHAND', False) and (
                not isinstance(epoch_samples, int) or epoch_samples <= 0):
            raise ValueError(
                'Paper mode requires DATASETS.TRAIN_SAMPLES_PER_EPOCH_PER_WORKER. '
                'Choose it from the dataset manifest, world size and worker count; '
                'the paper does not publish an exact sampling schedule.'
            )
        log.warning(
            'Paper mode: 1000 epochs. FreiHAND-only finite sampling is a local '
            'adaptation; official mixture weights are published, but epoch length is not. '
            'Author reference: 8 GPUs, effective batch 1024 (GitHub issue #19).'
        )

    # Load dataset config
    dataset_cfg = dataset_config(cfg.get('dataset_config', 'datasets_tar.yaml'))

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = WiLoRDataModule(cfg, dataset_cfg)

    # Setup model
    model = WiLoR(cfg, init_renderer=(cfg.GENERAL.get('ENABLE_VISUALIZATION', True)
                                    and cfg.MODEL.get('PROJECTION', 'perspective') != 'weak_perspective'))

    # Setup Tensorboard logger
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    rich_callback = pl.callbacks.RichProgressBar()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback,
        lr_monitor,
        # rich_callback
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train the model
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.get('ckpt_path'),
        # Full training checkpoints contain optimizer state and OmegaConf objects.
        # Only resume checkpoints produced by a trusted source.
        weights_only=False,
    )
    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path=str(root/"configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
