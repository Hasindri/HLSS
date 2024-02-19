"""HiDisc training script.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import Dict, Any
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

import pytorch_lightning as pl
# from pytorch_lightning.utilities.cloud_io import load as pl_load
import torchmetrics

from models import MLP, VisGranularNetwork, HLSSHidiscNetwork,resnet_backbone, ContrastiveLearningNetwork, HLSSContrastiveLearningNetwork, CLIPTextClassifier,CLIPVisual,HLSSKL
from common import (setup_output_dirs, parse_args, get_exp_name,
                           config_loggers, get_optimizer_func,
                           get_scheduler_func, get_dataloaders)
from losses.hidisc import HiDiscLoss

from clip.clip import load

# from eval_knn_hlss import *

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

        if cf["model"]["backbone"] == "RN50":
            bb = partial(CLIPVisual, arch=cf["model"]["backbone"])
            # bb = CLIPVisual(arch=cf["model"]["backbone"])
        else:
            raise NotImplementedError()

        mlp1 = partial(CLIPTextClassifier,n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"],
                      arch=cf["model"]["backbone"],
                      labels = cf["data"]["patch"],
                      templates=cf["model"]["patch_templates"])
        mlp2 = partial(CLIPTextClassifier,n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"],
                      arch=cf["model"]["backbone"],
                      labels = cf["data"]["slide"],
                      templates=cf["model"]["slide_templates"])
        mlp3 = partial(CLIPTextClassifier,n_in=1024,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"],
                      arch=cf["model"]["backbone"],
                      labels = cf["data"]["patient"],
                      templates=cf["model"]["patient_templates"])

        
        self.model = HLSSKL(bb, mlp1,mlp2,mlp3)
        self.patch_emb = torch.transpose(self.model.proj1.zeroshot_weights,1,0).to("cuda")
        self.slide_emb = torch.transpose(self.model.proj2.zeroshot_weights,1,0).to("cuda")
        self.patient_emb = torch.transpose(self.model.proj3.zeroshot_weights,1,0).to("cuda")

        if self.freeze_mlp:
            for param in self.model.proj1.parameters():
                param.requires_grad = False
            for param in self.model.proj2.parameters():
                param.requires_grad = False
            for param in self.model.proj3.parameters():
                param.requires_grad = False
  

        if "training" in cf:
            crit_params = cf["training"]["objective"]["params"]

            #granulartxt
            self.criterion1 = HiDiscLoss(
                lambda_patient=0,
                lambda_slide=0,
                lambda_patch=crit_params["lambda_patch"],
                supcon_loss_params=crit_params["supcon_params"])
            self.criterion2 = HiDiscLoss(
                lambda_patient=0,
                lambda_slide=crit_params["lambda_slide"],
                lambda_patch=0,
                supcon_loss_params=crit_params["supcon_params"])
            self.criterion3 = HiDiscLoss(
                lambda_patient=crit_params["lambda_patient"],
                lambda_slide=0,
                lambda_patch=0,
                supcon_loss_params=crit_params["supcon_params"])

            self.train_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "patient_kl": torchmetrics.MeanMetric(),
                "slide_kl": torchmetrics.MeanMetric(),
                "patch_kl": torchmetrics.MeanMetric(),
                "sum_loss": torchmetrics.MeanMetric()
            }) # yapf: disable
            self.val_loss = torch.nn.ModuleDict({
                "patient_loss": torchmetrics.MeanMetric(),
                "slide_loss": torchmetrics.MeanMetric(),
                "patch_loss": torchmetrics.MeanMetric(),
                "patient_kl": torchmetrics.MeanMetric(),
                "slide_kl": torchmetrics.MeanMetric(),
                "patch_kl": torchmetrics.MeanMetric(),
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
        bb_out,pred1, pred2, pred3 = self.model(im_reshaped)

        bb_out = bb_out.reshape(*batch["image"].shape[:4], bb_out.shape[-1])
        pred1 = pred1.reshape(*batch["image"].shape[:4], pred1.shape[-1])
        pred2 = pred2.reshape(*batch["image"].shape[:4], pred2.shape[-1])
        pred3 = pred3.reshape(*batch["image"].shape[:4], pred3.shape[-1])

        bb_gather = self.all_gather(bb_out, sync_grads=True)
        bb_gather = bb_gather.reshape(-1, *bb_gather.shape[2:])
        bb_gather_softmax = F.softmax(bb_gather, dim=-1)
        bsz = bb_gather.shape
        
        pred_gather1 = self.all_gather(pred1, sync_grads=True)
        pred_gather1 = pred_gather1.reshape(-1, *pred_gather1.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)
        patch_loss = self.criterion1(pred_gather1, label_gather)
   
        pred_gather2 = self.all_gather(pred2, sync_grads=True)
        pred_gather2 = pred_gather2.reshape(-1, *pred_gather2.shape[2:])
        slide_loss = self.criterion2(pred_gather2, label_gather)

        pred_gather3 = self.all_gather(pred3, sync_grads=True)
        pred_gather3 = pred_gather3.reshape(-1, *pred_gather3.shape[2:])
        patient_loss = self.criterion3(pred_gather3, label_gather)

        #contrastive losses
        losses = {key: patch_loss[key] + slide_loss[key] + patient_loss[key] for key in ['patch_loss','slide_loss','patient_loss']}

        # KL loss
        bb_gather1 = bb_gather.clone().unsqueeze(-2)
        patch_emb_reshaped = self.patch_emb.view(1, 1, 1, 1, self.patch_emb.size(0), self.patch_emb.size(1))
        cos_sim = F.cosine_similarity(bb_gather1, patch_emb_reshaped, dim=-1)
        ind = torch.argmax(cos_sim, dim=-1).view(-1)
        patch_attr = self.patch_emb[ind]
        patch_attr = patch_attr.view(bsz)

        slide_emb_reshaped = self.slide_emb.view(1, 1, 1, 1, self.slide_emb.size(0), self.slide_emb.size(1))
        cos_sim = F.cosine_similarity(bb_gather1, slide_emb_reshaped, dim=-1)
        ind = torch.argmax(cos_sim, dim=-1).view(-1)
        slide_attr = self.slide_emb[ind]
        slide_attr = slide_attr.view(bsz)

        patient_emb_reshaped = self.patient_emb.view(1, 1, 1, 1, self.patient_emb.size(0), self.patient_emb.size(1))
        cos_sim = F.cosine_similarity(bb_gather1, patient_emb_reshaped, dim=-1)
        ind = torch.argmax(cos_sim, dim=-1).view(-1)
        patient_attr = self.patient_emb[ind]
        patient_attr = patient_attr.view(bsz)

        patch_attr_softmax = F.softmax(patch_attr, dim=-1)
        patch_kl = (F.kl_div(torch.log(patch_attr_softmax), bb_gather_softmax, reduction='sum').item() + F.kl_div(torch.log(bb_gather_softmax), patch_attr_softmax, reduction='sum').item())/2
        slide_attr_softmax = F.softmax(slide_attr, dim=-1)
        slide_kl = (F.kl_div(torch.log(slide_attr_softmax), bb_gather_softmax, reduction='sum').item() + F.kl_div(torch.log(bb_gather_softmax), slide_attr_softmax, reduction='sum').item())/2
        patient_attr_softmax = F.softmax(patient_attr, dim=-1)
        patient_kl = (F.kl_div(torch.log(patient_attr_softmax), bb_gather_softmax, reduction='sum').item() + F.kl_div(torch.log(bb_gather_softmax), patient_attr_softmax, reduction='sum').item())/2

        #if not considering KL
        # patch_kl = 0
        # slide_kl = 0
        # patient_kl = 0

        #KL losses
        losses['patch_kl'] = patch_kl
        losses['slide_kl'] = slide_kl
        losses['patient_kl'] = patient_kl
        losses['sum_loss'] = patch_loss['sum_loss'] + slide_loss['sum_loss'] + patient_loss['sum_loss'] + patch_kl + slide_kl + patient_kl

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
        bb_out,pred1, pred2, pred3 = self.model(im_reshaped)

        bb_out = bb_out.reshape(*batch["image"].shape[:4], bb_out.shape[-1])
        pred1 = pred1.reshape(*batch["image"].shape[:4], pred1.shape[-1])
        pred2 = pred2.reshape(*batch["image"].shape[:4], pred2.shape[-1])
        pred3 = pred3.reshape(*batch["image"].shape[:4], pred3.shape[-1])

        bb_gather = self.all_gather(bb_out, sync_grads=True)
        bb_gather = bb_gather.reshape(-1, *bb_gather.shape[2:])
        bb_gather_softmax = F.softmax(bb_gather, dim=-1)
        bsz = bb_gather.shape
        
        pred_gather1 = self.all_gather(pred1, sync_grads=True)
        pred_gather1 = pred_gather1.reshape(-1, *pred_gather1.shape[2:])
        label_gather = self.all_gather(batch["label"]).reshape(-1, 1)
        patch_loss = self.criterion1(pred_gather1, label_gather)
   
        pred_gather2 = self.all_gather(pred2, sync_grads=True)
        pred_gather2 = pred_gather2.reshape(-1, *pred_gather2.shape[2:])
        slide_loss = self.criterion2(pred_gather2, label_gather)

        pred_gather3 = self.all_gather(pred3, sync_grads=True)
        pred_gather3 = pred_gather3.reshape(-1, *pred_gather3.shape[2:])
        patient_loss = self.criterion3(pred_gather3, label_gather)

        #contrastive losses
        losses = {key: patch_loss[key] + slide_loss[key] + patient_loss[key] for key in ['patch_loss','slide_loss','patient_loss']}

        # KL loss
        bb_gather1 = bb_gather.clone().unsqueeze(-2)
        patch_emb_reshaped = self.patch_emb.view(1, 1, 1, 1, self.patch_emb.size(0), self.patch_emb.size(1))
        cos_sim = F.cosine_similarity(bb_gather1, patch_emb_reshaped, dim=-1)
        ind = torch.argmax(cos_sim, dim=-1).view(-1)
        patch_attr = self.patch_emb[ind]
        patch_attr = patch_attr.view(bsz)

        slide_emb_reshaped = self.slide_emb.view(1, 1, 1, 1, self.slide_emb.size(0), self.slide_emb.size(1))
        cos_sim = F.cosine_similarity(bb_gather1, slide_emb_reshaped, dim=-1)
        ind = torch.argmax(cos_sim, dim=-1).view(-1)
        slide_attr = self.slide_emb[ind]
        slide_attr = slide_attr.view(bsz)

        patient_emb_reshaped = self.patient_emb.view(1, 1, 1, 1, self.patient_emb.size(0), self.patient_emb.size(1))
        cos_sim = F.cosine_similarity(bb_gather1, patient_emb_reshaped, dim=-1)
        ind = torch.argmax(cos_sim, dim=-1).view(-1)
        patient_attr = self.patient_emb[ind]
        patient_attr = patient_attr.view(bsz)

        patch_attr_softmax = F.softmax(patch_attr, dim=-1)
        patch_kl = (F.kl_div(torch.log(patch_attr_softmax), bb_gather_softmax, reduction='sum').item() + F.kl_div(torch.log(bb_gather_softmax), patch_attr_softmax, reduction='sum').item())/2
        slide_attr_softmax = F.softmax(slide_attr, dim=-1)
        slide_kl = (F.kl_div(torch.log(slide_attr_softmax), bb_gather_softmax, reduction='sum').item() + F.kl_div(torch.log(bb_gather_softmax), slide_attr_softmax, reduction='sum').item())/2
        patient_attr_softmax = F.softmax(patient_attr, dim=-1)
        patient_kl = (F.kl_div(torch.log(patient_attr_softmax), bb_gather_softmax, reduction='sum').item() + F.kl_div(torch.log(bb_gather_softmax), patient_attr_softmax, reduction='sum').item())/2

        #if not considering KL
        # patch_kl = 0
        # slide_kl = 0
        # patient_kl = 0

        #KL losses
        losses['patch_kl'] = patch_kl
        losses['slide_kl'] = slide_kl
        losses['patient_kl'] = patient_kl
        losses['sum_loss'] = patch_loss['sum_loss'] + slide_loss['sum_loss'] + patient_loss['sum_loss'] + patch_kl + slide_kl + patient_kl

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
        opt = get_optimizer_func(self.cf_)(self.model.parameters())

        # print(f' param {list(self.model.proj.parameters())}')

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
        max_epochs=cf["training"]["num_epochs"],
        check_val_every_n_epoch=cf["training"]["eval_ckpt_ep_freq"],
        num_nodes=1)
    
    exp = HiDiscSystem(cf, num_it_per_ep,freeze_mlp=False)

    # trainer.fit(exp,
    #             train_dataloaders=train_loader,
    #             val_dataloaders=valid_loader, ckpt_path = "/data1/dri/hidisc/hidisc/exps/Exp016/a/6b8280e8-Jan29-02-00-19-patient_disc_dev_/models/ckpt-epoch19999.ckpt")


    trainer.fit(exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
    main()
