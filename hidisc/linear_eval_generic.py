"""Evaluation modules and script.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any
import tifffile

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torchvision.transforms import Compose
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy

from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug,get_srh_base_aug_hidisc
from common import (parse_args, get_exp_name, config_loggers,
                           get_num_worker)
from train_hlss_128out import HiDiscSystem
# from train_hidisc_128out import HiDiscSystem
from models import MLP

import wandb

wandb.init(project="HLSS")


def get_embeddings(cf: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    # get model
    if cf["model"]["backbone"] == "RN50":
        aug_func = get_srh_base_aug
    elif cf["model"]["backbone"] == "resnet50":
        aug_func = get_srh_base_aug_hidisc
    elif cf["model"]["backbone"] == "vit":
        aug_func = get_srh_vit_base_aug
    else:
        raise NotImplementedError()

    # get dataset / loader
    train_dset = OpenSRHDataset(data_root=cf["data"]["db_root"],
                                studies="train",
                                transform=Compose(aug_func()),
                                balance_patch_per_class=False)
    train_dset.reset_index()


    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True)

    val_dset = OpenSRHDataset(data_root=cf["data"]["db_root"],
                              studies="val",
                              transform=Compose(aug_func()),
                              balance_patch_per_class=False)
    val_dset.reset_index()


    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True)

    # load lightning checkpoint
    ckpt_path = cf["eval"]["ckpt_path"]
    print(ckpt_path)

    model = HiDiscSystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0,
                                              max_epochs=-1,
                                              nc=0, freeze_mlp=True)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=-1,
                         default_root_dir=cf["infra"]["log_dir"],
                         enable_checkpointing=False,
                         logger=False)

    # generate predictions
    train_predictions = trainer.predict(model, dataloaders=train_loader)
    val_predictions = trainer.predict(model, dataloaders=val_loader)
    del model

    def process_predictions(predictions):
        pred = {}
        for k in predictions[0].keys():
            if k == "path":
                pred[k] = [pk for p in predictions for pk in p[k][0]]
            else:
                pred[k] = torch.cat([p[k] for p in predictions])
        return pred

    train_predictions = process_predictions(train_predictions)
    val_predictions = process_predictions(val_predictions)

    train_out = train_predictions["embeddings"]
    train_out = train_out.squeeze(1)
    train_label = train_predictions["label"]
    val_out = val_predictions["embeddings"]
    val_out = val_out.squeeze(1)
    val_label = val_predictions["label"]

    train_out = train_out.cpu().detach().numpy()
    train_label = train_label.cpu().detach().numpy()
    val_out = val_out.cpu().detach().numpy()
    val_label = val_label.cpu().detach().numpy()

    # Calculate mean embeddings for each class
    num_classes = 7
    train_mean_embeddings = [np.mean(train_out[train_label.flatten() == i], axis=0) for i in range(num_classes)]
    val_mean_embeddings = [np.mean(val_out[val_label.flatten() == i], axis=0) for i in range(num_classes)]

    # Plotting
    # fig, axes = plt.subplots(num_classes, 1, figsize=(12, 3 * num_classes))
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # breakpoint()
    attr = list(cf["model"]["templates"].keys())
    breakpoint()

    for i in tqdm(range(1)):
        plt.figure()
        train_mean_embeddings[i] = torch.tensor(train_mean_embeddings[i])
        original = train_mean_embeddings[i]
        mask = torch.abs(original) > 0.1
        sel_ind = np.where(mask)[0]
        sel_attr = [attr[idx] for idx in sel_ind]
        filtered = torch.zeros_like(original)
        filtered[mask] = original[mask]
        breakpoint()
        
        # sns.lineplot(x=np.arange(128), y=filtered_tensor, ax=axes[i])
        plt.bar(sel_attr, filtered[mask], width=0.8, color='blue', alpha=0.7)
        plt.title(f'{train_loader.dataset.classes_[i]}')

        plt.savefig(f'Exp007_{train_loader.dataset.classes_[i]}.png')
        plt.show()

    # wandb.log({'embedding_plots': wandb.Image('mean_embeddings_plot2.png')})

    plt.close()

    wandb.finish()

    return val_predictions, train_mean_embeddings



def main():
    """Driver script for evaluation pipeline."""
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    pl.seed_everything(cf["infra"]["seed"])


    val_predictions, train_mean_embeddings = get_embeddings(cf)

    # # train_mean_embeddings = torch.stack(train_mean_embeddings, dim=1)
    # tensor_list = [torch.tensor(arr) for arr in train_mean_embeddings]
    # train_mean_embeddings = torch.stack(tensor_list, dim=1)

    # mlp_model = MLP(n_in=128, hidden_layers=[], n_out=7)

    # with torch.no_grad():
    #     mlp_model.layers[0].weight.copy_(train_mean_embeddings.t())
    #     mlp_model.layers[0].bias.fill_(0.01)

    # input_embeddings = val_predictions["embeddings"]
    # input_embeddings = input_embeddings.squeeze(dim=1)
    # logits = mlp_model(input_embeddings)
    # probs = F.softmax(logits, dim=1)
    # predicted_class = torch.argmax(probs, dim=1)
    # true_labels = val_predictions["label"]
    # loss = F.cross_entropy(logits, true_labels)
    
    # print("Cross-Entropy Loss:", loss.item())

    # c=0
    # for i in range (true_labels.shape[0]):
    #     if predicted_class[i] == true_labels[i]:
    #         c+=1
    # acc = (c/true_labels.shape[0])*100
    # print(f'linear classification acc: {acc}')

if __name__ == "__main__":
    main()
