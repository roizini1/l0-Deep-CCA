from omegaconf import OmegaConf
import logging
import torch

# pytorch lightning imports:
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

# hydra imports:
import hydra
from hydra.core.config_store import ConfigStore

#  relevant files imports:
from pl_net import Net
from config_classes import Config_class
from _loader import MNISTDataModule

# empty gpu cache:
torch.cuda.empty_cache()

# config store for hydra:
cs = ConfigStore.instance()
cs.store(name="Config", node=Config_class)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: Config_class):
    logger = logging.getLogger(__name__)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    model = Net(cfg)

    checkpoint_callback = ModelCheckpoint(
        filename='checkpoint_{epoch:02d}-{loss:.2f}',
        monitor='loss',
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode='min',
    )

    stop_callback = EarlyStopping(
        monitor='loss',
        patience=30,
        mode='min',  # for this loss mode is max
        )

    callback_list = [checkpoint_callback, RichProgressBar(), stop_callback, ]

    tb_logger = TensorBoardLogger(save_dir="/home/dsi/ziniroi/project_usl/src/")

    my_strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)

    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        max_epochs=cfg.max_epochs,
        enable_checkpointing=True,
        callbacks=callback_list,
        logger=tb_logger,
        strategy=my_strategy,
        sync_batchnorm=True,
        detect_anomaly=True,
    )
    '''
    
    # single gpu
    trainer = Trainer(
        # deterministic=True,
        accelerator="gpu",

        devices=[1],
        max_epochs=cfg.max_epochs,
        enable_checkpointing=True,
        #  enable_model_summary=True,
        callbacks=callback_list,
        logger=tb_logger,
        detect_anomaly=True,
        # strategy=my_strategy,
        # strategy="ddp",
        # sync_batchnorm=True,
        # auto_lr_find=True,
        # auto_scale_batch_size="binsearch",
    )
    '''
    dm = MNISTDataModule()
    dm.setup(stage="fit")
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
