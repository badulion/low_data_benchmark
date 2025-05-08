#######################################################
# General start script for all models based on config #
#######################################################

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
import signal, sys

from cProfile import Profile
from pstats import SortKey, Stats

from torch.nn.functional import mse_loss, rms_norm, l1_loss
from torch.optim import Adam

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader, random_split
from dynabench.dataset.transforms import Compose

class DynabenchDataModule(LightningDataModule):
    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.Batch_size
        self.num_workers = cfg.Workers

        self.transforms = Compose([instantiate(t) for t in cfg.transforms])

        self.train_rollout = cfg.TrainRollout
        self.val_rollout = cfg.TestRollout
        self.test_rollout = cfg.TestRollout

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if self.cfg.Resolution == 'full':
            if stage == "fit" or stage is None:
                self.train_dataset = instantiate(self.cfg.dbiterator,
                    split="train",
                    structure='grid',
                    rollout=self.train_rollout,
                    transforms=self.transforms,
                )
                self.val_dataset = instantiate(self.cfg.dbiterator,
                    split="val",
                    structure='grid',
                    rollout=self.val_rollout,
                    transforms=self.transforms,
                )
            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_dataset = instantiate(self.cfg.dbiterator,
                    split="test",
                    structure='grid',
                    rollout=self.test_rollout,
                    transforms=self.transforms,
                )
        else:
            if stage == "fit" or stage is None:
                self.train_dataset = instantiate(self.cfg.dbiterator,
                    split="train",
                    structure=self.cfg.Structure,
                    rollout=self.train_rollout,
                    transforms=self.transforms,
                )
                self.val_dataset = instantiate(self.cfg.dbiterator,
                    split="val",
                    structure=self.cfg.Structure,
                    rollout=self.val_rollout,
                    transforms=self.transforms,
                )
            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_dataset = instantiate(self.cfg.dbiterator,
                    split="test",
                    structure=self.cfg.Structure,
                    rollout=self.test_rollout,
                    transforms=self.transforms,
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
                 type,
                 device,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.pars = hparams
        self.model = model
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.structure = structure
        self._type = type
        self._device = device
        
        summary = summarize(self, max_depth=-1)
        trainable_params = summary.trainable_parameters

        self.save_hyperparameters({"trainable_parameters": trainable_params})
        self.save_hyperparameters('hparams')

        self._start = None
        self._end = None
    
    def _step(self, batch):
        
        # extract number of steps
        x = batch['x']
        y = batch['y']
        p = batch['pos']
        if 'knn_graph' in batch.keys():
            edge_index = batch['knn_graph']
        else:
            edge_index = None

        #x = x[:,0]
        steps = y.shape[1]
        # prediction steps
        t = torch.arange(0, steps, device=x.device)
        if self._type == 'grid':
            pred = self.model(x=x, t_eval=t)
        elif self._type == 'point':
            pred = self.model(x=x, p=p, t_eval=t)
        elif self._type == 'graph':
            pred = self.model(x=x, p=p, edge_index=edge_index, t_eval=t)
        else:
            pred = x
        loss = self.lossfunc(pred[:,-1], y[:,-1])
        loss = []
        for i in range(pred.shape[1]):
            loss.append(self.lossfunc(pred[:,i], y[:,i]))

        loss_mse = loss[-1]  # Assuming loss[0] is mse
        loss_l1 = l1_loss(pred[:,-1], y[:,-1])
        loss_rmse = torch.sqrt(loss_mse)

        return loss, loss_mse, loss_l1, loss_rmse, pred
    
    def training_step(self, batch, batch_idx):
        if self._device == 'gpu': self._start = torch.cuda.Event(enable_timing=True)
        if self._device == 'gpu': self._end = torch.cuda.Event(enable_timing=True)

        if self._device == 'gpu': self._start.record()
        loss, loss_mse, loss_l1, loss_rmse, pred = self._step(batch)

        self.log('train_loss', loss_mse, prog_bar=True)
        self.log('train_loss_rmse', loss_rmse)
        self.log('train_loss_l1', loss_l1)
        return loss_mse

    def validation_step(self, batch, batch_idx):
        loss, loss_mse, loss_l1, loss_rmse, pred = self._step(batch)
        self.log("val_loss", loss_mse, prog_bar=True)
        self.log("val_loss_rmse", loss_rmse)
        self.log("val_loss_l1", loss_l1)
        return loss_mse
    
    def test_step(self, batch, batch_idx):
        if self._device == 'gpu': _start = torch.cuda.Event(enable_timing=True)
        if self._device == 'gpu': _end = torch.cuda.Event(enable_timing=True)

        if self._device == 'gpu': _start.record()
        loss, loss_mse, loss_l1, loss_rmse, pred = self._step(batch)
        if self._device == 'gpu': _end.record()
        if self._device == 'gpu': torch.cuda.synchronize()
        if self._device == 'gpu': elapsed_time = _start.elapsed_time(_end)
        if self._device == 'gpu': self.log('test_time', elapsed_time)
        self.log("test_loss", loss_mse)
        self.log("test_loss_rmse", loss_rmse)
        self.log("test_loss_l1", loss_l1)

        # log 1 step and 16 step loss seperatly
        for i in range(len(loss)):
            self.log(f'test_loss-{i+1}', loss[i])
        return loss_mse

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False
    ):
        optimizer.zero_grad()
        optimizer_closure()
        optimizer.step()

        # End timing after optimizer step
        if self._device == 'gpu': self._end.record()
        if self._device == 'gpu': torch.cuda.synchronize()
        if self._device == 'gpu': total_time_ms = self._start.elapsed_time(self._end)

        # Log total step time
        if self._device == 'gpu': self.log("train_step_ms", total_time_ms)

    def configure_optimizers(self):
        return self.optimizer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:

    # wandb_key = f"{os.getenv('WANDBKEY')}"
    wandb_key = "6813c95f67c8b605e828ee3aba39eeec4b0ebad4"
    wandbloggedin = wandb.login(key = wandb_key, relogin=True)

    print(cfg.model.name)
    print(cfg.resolution)

    # compose transforms according to model
    transforms = []
    if cfg.Structure == 'cloud' and cfg.resolution.name == 'full':
        transforms.append(cfg.transforms['Grid2Cloud'])
    if cfg.type == 'graph':
        transforms.append(cfg.transforms['EdgeList'])
    transforms.append(cfg.transforms['ToDict'])
    cfg.transforms = transforms
        
    

    cfg.InChannels = cfg.InChannels * cfg.lookback

    model = instantiate(cfg.wrapper)
    datamodule = DynabenchDataModule(cfg)
    
    # if baseline -> perform only forward pass and mean
    if 'zero' in cfg.model.name or 'persistance' in cfg.model.name:
        # skip training
        # test model
        optimizer = None
        pl_model = pl_wrapper(hparams=cfg, hparams_model=cfg.model, model=model, lossfunc=mse_loss, optimizer=optimizer, structure=cfg.Structure, type=cfg.type, device=cfg.device)
        trainer = instantiate(cfg.trainer)
        trainer.test(datamodule=datamodule, model=pl_model)
    else:
        optimizer = Adam(model.parameters(), lr=cfg.LearningRate, weight_decay=cfg.WeightDecay)
        pl_model = pl_wrapper(hparams=cfg, hparams_model=cfg.model, model=model, lossfunc=mse_loss, optimizer=optimizer, structure=cfg.Structure, type=cfg.type, device=cfg.device)
        trainer = instantiate(cfg.trainer)
        # train model
        trainer.fit(model=pl_model, datamodule=datamodule, ckpt_path="last")
        # test model
        trainer.test(datamodule=datamodule, ckpt_path='best')

    with open(f'{cfg.output_dir}/status/{cfg.version_name}.txt', 'w') as f:
        f.write('TRAINING_COMPLETED')
    
    wandb.finish()


def handler(signum, frame):
    print("Received SIGUSR1, interrupting training...")
    wandb.finish()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, handler)
    main()