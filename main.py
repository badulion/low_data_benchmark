#######################################################
# General start script for all models based on config #
#######################################################

## Train the Hypersolver + NeuralPDE
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from typing import Any
import os
import sys

from cProfile import Profile
from pstats import SortKey, Stats

from torch.nn.functional import mse_loss
from torch.optim import Adam

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class DynabenchDataset(torch.utils.data.Dataset):
    def __init__( 
            self,
            cfg,
            split: str="train",
            rollout: int = 16,
            **kwargs):
        self.iterator = instantiate(cfg.dbiterator, split=split, rollout=rollout)

    def __len__(self):
        return len(self.iterator)
    
    def __getitem__(self, index):
        x, y, p = self.iterator[index]
        return x.astype(np.float32), y.astype(np.float32), p.astype(np.float32)

class DynabenchDataModule(LightningDataModule):
    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.Batch_size
        self.num_workers = cfg.Workers

        self.train_rollout = cfg.TrainRollout
        self.val_rollout = cfg.TestRollout
        self.test_rollout = cfg.TestRollout

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = DynabenchDataset(
                split="train",
                rollout=self.train_rollout,
                cfg=self.cfg
            )
            self.val_dataset = DynabenchDataset(
                split="val",
                rollout=self.val_rollout,
                cfg=self.cfg
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = DynabenchDataset(
                split="test",
                rollout=self.test_rollout,
                cfg=self.cfg
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class pl_wrapper(pl.LightningModule):
    def __init__(self,
                 hparams: dict,
                 hparams_model: dict,
                 model: nn.Module,
                 lossfunc,
                 optimizer,
                 structure,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters('hparams')
        self.save_hyperparameters('hparams_model')
        self.model = model
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.structure = structure

    def _step(self, batch):
        # extract number of steps
        x,y,p = batch
        #animate_simulation([y[0,i,0].cpu().numpy() for i in range(y.shape[1])])
        x = x[:,0]
        steps = y.shape[1]
        # prediction steps
        t = torch.arange(0, steps, device=batch[0].device)
        if self.structure == 'grid':
            pred = self.model(x=x, t_eval=t)
        elif self.structure == 'cloud':
            pred = self.model(x=x, p=p, t_eval=t)
        else:
            pred = x
        loss = self.lossfunc(pred[:,-1], y[:,-1])
        loss = []
        for i in range(pred.shape[1]):
            loss.append(self.lossfunc(pred[:,i], y[:,i]))

        return loss, pred
    
    def training_step(self, batch, batch_idx):
        loss, pred = self._step(batch)
        self.log('train_loss', loss[-1], prog_bar = True)
        return loss[-1]

    def validation_step(self, batch, batch_idx):
        val_loss, pred = self._step(batch)
        self.log("val_loss", val_loss[-1], prog_bar=True)
        return val_loss[-1]
    
    def test_step(self, batch, batch_idx):
        # wallclock time measuring of inference
        loss, pred = self._step(batch)
        # log 1 step and 16 step loss seperatly
        for i in range(len(loss)):
            self.log(f'test_loss-{i+1}', loss[i])
        return loss[-1]

    def configure_optimizers(self):
        return self.optimizer

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:

    #wandbloggedin = wandb.login(key = f"{os.getenv('WANDBKEY')}", relogin=True)

    model = instantiate(cfg.wrapper)
    datamodule = DynabenchDataModule(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.LearningRate, weight_decay=cfg.WeightDecay)
    pl_model = pl_wrapper(hparams=cfg, hparams_model=cfg.MODEL, model=model, lossfunc=mse_loss, optimizer=optimizer, structure=cfg.Structure)
    trainer = instantiate(cfg.trainer)
    
    # train model
    trainer.fit(model=pl_model, datamodule=datamodule, ckpt_path="last")
    # test model
    trainer.test(datamodule=datamodule, ckpt_path='best')
    
    wandb.finish()



if __name__ == "__main__":
    main()