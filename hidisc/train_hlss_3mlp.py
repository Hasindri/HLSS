"""HiDisc training script.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import List, Union, Dict, Any
from shutil import copy2
import os
import tifffile
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import torch
from torchvision.transforms import Compose
from torch.utils.data.dataset import Subset


import pytorch_lightning as pl
# from pytorch_lightning.utilities.cloud_io import load as pl_load
import torchmetrics

from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug_hidisc, get_srh_vit_base_aug, get_srh_base_aug
from models import MLP, resnet_backbone, ContrastiveLearningNetwork, three_MLP, CLIPVisual
from common import (setup_output_dirs, parse_args, get_exp_name,
                           config_loggers, get_optimizer_func,
                           get_scheduler_func, get_dataloaders, get_num_worker, get_dataloaders_hidisc)
from losses.hidisc import HiDiscLoss
# from eval_knn import *

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import wandb

wandb.init(project="HLSS")

class HiDiscSystem(pl.LightningModule):
    """Lightning system for hidisc experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int,freeze_mlp: bool):
        super().__init__()
        self.cf_ = cf
        self.freeze_mlp = freeze_mlp
        # self.exp_root = exp_root
        # self.pred_dir = os.path.join(exp_root,"predictions")

        if cf["model"]["backbone"] == "RN50":
            bb = partial(CLIPVisual, arch=cf["model"]["backbone"])
        else:
            raise NotImplementedError()

        mlp1 = partial(MLP,
                      n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        mlp2 = partial(MLP,
                      n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        mlp3 = partial(MLP,
                      n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
                      
        self.model = three_MLP(bb, mlp1, mlp2, mlp3)

        # print(f'model keys {self.model.proj}')

        if self.freeze_mlp:
            for param in self.model.proj1.parameters():
                param.requires_grad = False
            for param in self.model.proj2.parameters():
                param.requires_grad = False
            for param in self.model.proj3.parameters():
                param.requires_grad = False

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
        # print(f'batch {batch["image"].shape}')
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        # print(f'im_reshaped {im_reshaped.shape}')
        pred1, pred2, pred3 = self.model(im_reshaped)
        pred1 = pred1.reshape(*batch["image"].shape[:4], pred1.shape[-1])
        pred2 = pred2.reshape(*batch["image"].shape[:4], pred2.shape[-1])
        pred3 = pred3.reshape(*batch["image"].shape[:4], pred3.shape[-1])
        # print(f'pred {pred.shape}')

        pred_gather1 = self.all_gather(pred1, sync_grads=True)
        pred_gather2 = self.all_gather(pred2, sync_grads=True)
        pred_gather3 = self.all_gather(pred3, sync_grads=True)

        pred_gather1 = pred_gather1.reshape(-1, *pred_gather1.shape[2:])
        pred_gather2 = pred_gather2.reshape(-1, *pred_gather2.shape[2:])
        pred_gather3 = pred_gather3.reshape(-1, *pred_gather3.shape[2:])

        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)
        # print(f'pred_gather {pred_gather.shape}')
        # print(f'label_gather {label_gather.shape}')

        losses1 = self.criterion(pred_gather1, label_gather)
        losses2 = self.criterion(pred_gather2, label_gather)
        losses3 = self.criterion(pred_gather3, label_gather)
        # print(f'losses {losses}')

        losses = {}
        for key in losses1.keys():
            losses[key] = (losses1[key] + losses2[key] + losses3[key]) / 3

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
        # print(f'im_reshaped {im_reshaped.shape}')
        pred1, pred2, pred3 = self.model(im_reshaped)
        pred1 = pred1.reshape(*batch["image"].shape[:4], pred1.shape[-1])
        pred2 = pred2.reshape(*batch["image"].shape[:4], pred2.shape[-1])
        pred3 = pred3.reshape(*batch["image"].shape[:4], pred3.shape[-1])
        # print(f'pred {pred.shape}')

        pred_gather1 = self.all_gather(pred1, sync_grads=True)
        pred_gather2 = self.all_gather(pred2, sync_grads=True)
        pred_gather3 = self.all_gather(pred3, sync_grads=True)

        pred_gather1 = pred_gather1.reshape(-1, *pred_gather1.shape[2:])
        pred_gather2 = pred_gather2.reshape(-1, *pred_gather2.shape[2:])
        pred_gather3 = pred_gather3.reshape(-1, *pred_gather3.shape[2:])

        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)
        # print(f'pred_gather {pred_gather.shape}')
        # print(f'label_gather {label_gather.shape}')

        losses1 = self.criterion(pred_gather1, label_gather)
        losses2 = self.criterion(pred_gather2, label_gather)
        losses3 = self.criterion(pred_gather3, label_gather)
        # print(f'losses {losses}')

        losses = {}
        for key in losses1.keys():
            losses[key] = (losses1[key] + losses2[key] + losses3[key]) / 3

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
            wandb.log({f"train/{k}": train_loss_k})
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
            wandb.log({f"val/{k}": val_loss_k})
            self.val_loss[k].reset()

    def configure_optimizers(self):
        # if not training, no optimizer
        if "training" not in self.cf_:
            return None

        # get optimizer
        # opt = get_optimizer_func(self.cf_)(self.model.parameters())
        opt = get_optimizer_func(self.cf_)(filter(lambda p: p.requires_grad, self.model.parameters()))

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
    # _, pred_dir, _, pred_fname = setup_eval_paths(cf, get_exp_name, "")
    print(f'exp_root {exp_root}')
    print(f'model_dir {model_dir}')
    print(f'cp_config {cp_config}')
    # print(f'pred_dir {pred_dir}')
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
        max_epochs=cf["training"]["num_epochs"],
        check_val_every_n_epoch=cf["training"]["eval_ckpt_ep_freq"],
        num_nodes=1)
    
    exp = HiDiscSystem(cf, num_it_per_ep,freeze_mlp=False)
    
    # trainer.fit(exp,
    #             train_dataloaders=train_loader,
    #             val_dataloaders=valid_loader, ckpt_path = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/exps/Exp009/a/c5b75587-Nov24-12-17-33-patient_disc_dev_/models/ckpt-epoch13999.ckpt")


    trainer.fit(exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
    main()
