"""HiDisc training script.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import Dict, Any
import os

import torch

import pytorch_lightning as pl
# from pytorch_lightning.utilities.cloud_io import load as pl_load
import torchmetrics

from models import MLP, resnet_backbone, ContrastiveLearningNetwork
from common import (setup_output_dirs, parse_args, get_exp_name,
                           config_loggers, get_optimizer_func,
                           get_scheduler_func, get_dataloaders)
from losses.hidisc import HiDiscLoss

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class HiDiscSystem(pl.LightningModule):
    """Lightning system for hidisc experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["backbone"] == "resnet50":
            bb = partial(resnet_backbone, arch=cf["model"]["backbone"])
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = ContrastiveLearningNetwork(bb, mlp)

        if "training" in cf:
            crit_params = cf["training"]["objective"]["params"]
            self.criterion = HiDiscLoss(
                lambda_patient=crit_params["lambda_patient"],
                lambda_slide=crit_params["lambda_slide"],
                lambda_patch=crit_params["lambda_patch"],
                supcon_loss_params=crit_params["supcon_params"])
            self.train_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "sum_loss": torchmetrics.MeanMetric()
            }) # yapf: disable
            self.val_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "sum_loss": torchmetrics.MeanMetric()
            })  #yapf: disable
        else:
            self.criterion = self.train_loss = self.val_loss = None

        self.num_it_per_ep_ = num_it_per_ep

    def forward(self, batch):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        return pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

    def training_step(self, batch, _):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        pred = pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        losses = self.criterion(pred_gather, label_gather)

        bs = batch["image"][0].shape[0] * torch.cuda.device_count()
        log_partial = partial(self.log,
                              on_step=True,
                              on_epoch=True,
                              batch_size=bs,
                              sync_dist=True,
                              rank_zero_only=True)
        for k in self.train_loss:
            log_partial(f"train/{k}", losses[k])
            self.train_loss[k].update(losses[k], weight=bs)

        return losses["sum_loss"]

    def validation_step(self, batch, batch_idx):
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        pred = self.model(im_reshaped)
        pred = pred.reshape(*batch["image"].shape[:4], pred.shape[-1])

        pred_gather = self.all_gather(pred, sync_grads=True)
        pred_gather = pred_gather.reshape(-1, *pred_gather.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)

        losses = self.criterion(pred_gather, label_gather)

        bs = batch["image"][0].shape[0] * torch.cuda.device_count()
        for k in self.val_loss:
            self.val_loss[k].update(losses[k], weight=bs)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        self.model.eval()
        assert len(batch["image"].shape) == 4
        out = self.model.bb(batch["image"])
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }

    def on_train_epoch_end(self):
        for k in self.train_loss:
            train_loss_k = self.train_loss[k].compute()
            self.log(f"train/{k}_manualepoch",
                     train_loss_k,
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            logging.info(f"train/{k}_manualepoch {train_loss_k}")
            self.train_loss[k].reset()

    def on_validation_epoch_end(self):
        for k in self.val_loss:
            val_loss_k = self.val_loss[k].compute()
            self.log(f"val/{k}_manualepoch",
                     val_loss_k,
                     on_epoch=True,
                     sync_dist=True,
                     rank_zero_only=True)
            logging.info(f"val/{k}_manualepoch {val_loss_k}")
            self.val_loss[k].reset()

    def configure_optimizers(self):
        # if not training, no optimizer
        if "training" not in self.cf_:
            return None

        # get optimizer
        opt = get_optimizer_func(self.cf_)(self.model.parameters())

        # check if use a learn rate scheduler
        sched_func = get_scheduler_func(self.cf_, self.num_it_per_ep_)
        if not sched_func:
            return opt

        # get learn rate scheduler
        lr_scheduler_config = {
            "scheduler": sched_func(opt),
            "interval": "step",
            "frequency": 1,
            "name": "lr"
        }

        return [opt], lr_scheduler_config


def main():
    cf_fd = parse_args()
    # print (f'args {cf_fd}')
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    # print(f'exp_root {exp_root}')
    # print(f'model_dir {model_dir}')
    # print(f'cp_config {cp_config}')
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    train_loader, valid_loader = get_dataloaders(cf)
    system_func = HiDiscSystem

    logging.info(f"num devices: {torch.cuda.device_count()}")
    logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    num_it_per_ep = len(train_loader)
    print(f'noof train minibatches {num_it_per_ep}')
    print(f'noof val minibatches {len(valid_loader)}')
    if torch.cuda.device_count() > 1:
        num_it_per_ep //= torch.cuda.device_count()
        print(f'num_it_per_ep after distribution {num_it_per_ep}')

  


    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    epoch_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=-1,
        every_n_epochs=cf["training"]["eval_ckpt_ep_freq"],
        filename="ckpt-epoch{epoch}",
        auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    # create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        default_root_dir=exp_root,
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False,
                                           static_graph=True),
        logger=logger,
        log_every_n_steps=10,
        callbacks=[epoch_ckpt, lr_monitor],
        max_epochs=cf["training"]["num_epochs"] + 23600,
        check_val_every_n_epoch=cf["training"]["eval_ckpt_ep_freq"],
        num_nodes=1)
    
    exp = HiDiscSystem(cf, num_it_per_ep)
       
    # Load pre-trained checkpoint if it exists
    # ckpt_path = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/hidisc_train/8d0f44c5-Jul31-14-57-44-patient_disc_dev_/models/ckpt-epoch6799.ckpt"
    # if os.path.isfile(ckpt_path):
    #     print("Loading pre-trained checkpoint...")
    #     checkpoint = torch.load(ckpt_path)
    #     # print(f'checkpoint {checkpoint["optimizer_states"]["optimizer"]}')
        
    #     exp = HiDiscSystem.load_from_checkpoint(ckpt_path, cf=cf, num_it_per_ep=num_it_per_ep)

    # # Update optimizer and scheduler states from the checkpoint
    #     if 'optimizer_states' in checkpoint:
    #         trainer.optimizers[0].load_state_dict(checkpoint['optimizer_states'][0])
    #     if 'lr_schedulers' in checkpoint:
    #         trainer.lr_schedulers[0]['scheduler'].load_state_dict(checkpoint['lr_schedulers'][0]['scheduler'])
    # else:
    #     exp = HiDiscSystem(cf, num_it_per_ep)


    # trainer.fit(exp,
    #             train_dataloaders=train_loader,
    #             val_dataloaders=valid_loader, ckpt_path = "/data1/dri/hidisc/hidisc/datasets/opensrh/hidisc_train/6edee2ae-Sep21-14-17-42-patch_disc_weak_/models/ckpt-epoch5599.ckpt")


    trainer.fit(exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
    main()
