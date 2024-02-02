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

import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy

from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug
from common import (parse_args, get_exp_name, config_loggers,
                           get_num_worker)
from train_hlss_128out import HiDiscSystem

# import wandb

# wandb.init(project="HLSS")


def get_embeddings(cf: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    # get model
    if cf["model"]["backbone"] == "RN50":
        aug_func = get_srh_base_aug
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
    ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_path"])
    print(ckpt_path)

    model = HiDiscSystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0,
                                              max_epochs=-1,
                                              nc=0, freeze_mlp=False)

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

    # print(f'train_predictions embeddings {train_predictions["embeddings"].shape}')
    # print(f'train_predictions labels {train_predictions["label"].shape}')

    # print(f'val_predictions embeddings {val_predictions["embeddings"].shape}')
    # print(f'val_predictions labels {val_predictions["label"].shape}')

    train_out = train_predictions["embeddings"]
    train_out = train_out.squeeze(1)
    print(f'train_out {train_out.shape}')
    train_label = train_predictions["label"]
    val_out = val_predictions["embeddings"]
    val_out = val_out.squeeze(1)
    print(f'val_out {val_out.shape}')
    val_label = val_predictions["label"]

    train_out = train_out.cpu().detach().numpy()
    print(f'train_out {train_out.shape}')
    train_label = train_label.cpu().detach().numpy()
    val_out = val_out.cpu().detach().numpy()
    print(f'val_out {val_out.shape}')
    val_label = val_label.cpu().detach().numpy()

    # Calculate mean embeddings for each class
    num_classes = 7
    train_mean_embeddings = [np.mean(train_out[train_label.flatten() == i], axis=0) for i in range(num_classes)]
    val_mean_embeddings = [np.mean(val_out[val_label.flatten() == i], axis=0) for i in range(num_classes)]
    print(f'train_mean_embeddings  {len(train_mean_embeddings)} , {train_mean_embeddings[0].shape}')
    print(f'val_mean_embeddings  {len(val_mean_embeddings)} , {val_mean_embeddings[0].shape}')

    # Plotting
    fig, axes = plt.subplots(num_classes, 2, figsize=(12, 3 * num_classes))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(num_classes):
        # Plot Training Mean Embedding
        values_to_plot = torch.where(train_mean_embeddings[i] > 0.1, train_mean_embeddings[i], 0)
        sns.lineplot(x=np.arange(128), y=values_to_plot, ax=axes[i, 0])
        axes[i, 0].set_title(f'Training Class {i}')

        # Plot Validation Mean Embedding
        values_to_plot = torch.where(val_mean_embeddings[i] > 0.1, val_mean_embeddings[i], 0)
        sns.lineplot(x=np.arange(128), y=values_to_plot, ax=axes[i, 1])
        axes[i, 1].set_title(f'Validation Class {i}')

    # Save Plot
    plt.savefig('mean_embeddings_plot.png')
    plt.show()


    plt.figure(figsize=(10, 5))

    for i in range(num_classes):
        # Plot Bar Plot with values > 0.1
        val_mean_embedding_tensor = torch.tensor(val_mean_embeddings[i])
        values_to_plot = torch.where(val_mean_embeddings[i] > 0.1, val_mean_embeddings[i], 0)
        sns.lineplot(x=np.arange(128), y=values_to_plot, label=f'Class {i}')
        
    # Set plot labels and legend
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Mean Embedding Value')
    plt.title('Validation Mean Embeddings')
    plt.legend()

    plt.savefig('mean_embeddings_plot2.png')

    wandb.log({'embedding_plots': wandb.Image('mean_embeddings_plot2.png')})

    plt.close()

    wandb.finish()




    return val_predictions



def main():
    """Driver script for evaluation pipeline."""
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    pl.seed_everything(cf["infra"]["seed"])


    predictions = get_embeddings(cf)

if __name__ == "__main__":
    main()
