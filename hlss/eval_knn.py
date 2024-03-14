"""Evaluation modules and script.

Copyright (c) 2024 Mohamed Bin Zayed University of Artificial Intelligence. All rights reserved.
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
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from torchvision.transforms import Compose
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy
from torchmetrics.classification import MulticlassAccuracy
from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug, get_srh_base_aug_hidisc
from common import (parse_args, get_exp_name, config_loggers,
                           get_num_worker)
from train_hidisc import HiDiscSystem

import wandb

wandb.init(project="HLSS")

# code for kNN prediction is from the github repo IgorSusmelj/barlowtwins
# https://github.com/IgorSusmelj/barlowtwins/blob/main/utils.py
def knn_predict(feature, feature_bank, feature_labels, classes: int,
                knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features from a feature bank.

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: Temperature
    """
    # cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1,
                              index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) *
                            sim_weight.unsqueeze(dim=-1),
                            dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels, pred_scores


def get_embeddings(cf: Dict[str, Any],
                   exp_root: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    # get model
    if cf["model"]["backbone"] == "resnet50":
        aug_func = get_srh_base_aug_hidisc
    elif cf["model"]["backbone"] == "RN50":
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
    # ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
    #                          cf["eval"]["ckpt_path"])
    
    ckpt_path = cf["eval"]["ckpt_path"]

    model = HiDiscSystem(cf=cf, num_it_per_ep=0,freeze_mlp=True)
    
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=-1,
                         default_root_dir=exp_root,
                         enable_checkpointing=False,
                         logger=False)


    # model = HiDiscSystem.load_from_checkpoint(ckpt_path,
    #                                           cf=cf,
    #                                           num_it_per_ep=0,
    #                                           max_epochs=-1,
    #                                           nc=0)

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

    train_embs = torch.nn.functional.normalize(train_predictions["embeddings"],
                                               p=2,
                                               dim=1).T
    val_embs = torch.nn.functional.normalize(val_predictions["embeddings"],
                                             p=2,
                                             dim=1)

    # knn evaluation
    batch_size = cf["eval"]["knn_batch_size"]
    all_scores = []
    for k in tqdm(range(val_embs.shape[0] // batch_size + 1)):
        start_coeff = batch_size * k
        end_coeff = min(batch_size * (k + 1), val_embs.shape[0])
        val_embs_k = val_embs[start_coeff:end_coeff]  # 1536 x 2048

        pred_labels, pred_scores = knn_predict(
            val_embs_k,
            train_embs,
            train_predictions["label"],
            len(train_loader.dataset.classes_),
            knn_k=200,
            knn_t=0.07)

        all_scores.append(
            torch.nn.functional.normalize(pred_scores, p=1, dim=1))
        torch.cuda.empty_cache()

    val_predictions["logits"] = torch.vstack(all_scores)
    return val_predictions


def make_specs(predictions: Dict[str, Union[torch.Tensor, List[str]]]) -> None:
    """Compute all specs for an experiment"""

    # aggregate prediction into a dataframe
    pred = pd.DataFrame.from_dict({
        "path":
        predictions["path"],
        "labels": [l.item() for l in list(predictions["label"])],
        "logits": [l.tolist() for l in list(predictions["logits"])]
    })
    pred["logits"] = pred["logits"].apply(
        lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0))

    # add patient and slide info from patch paths
    pred["patient"] = pred["path"].apply(lambda x: x.split("/")[-4])
    pred["slide"] = pred["path"].apply(lambda x: "/".join(
        [x.split("/")[-4], x.split("/")[-3]]))

    # aggregate logits
    get_agged_logits = lambda pred, mode: pd.DataFrame(
        pred.groupby(by=[mode, "labels"])["logits"].apply(
            lambda x: [sum(y) for y in zip(*x)])).reset_index()

    slides = get_agged_logits(pred, "slide")
    patients = get_agged_logits(pred, "patient")

    normalize_f = lambda x: torch.nn.functional.normalize(x, dim=1, p=1)
    patch_logits = normalize_f(torch.tensor(np.vstack(pred["logits"])))
    slides_logits = normalize_f(torch.tensor(np.vstack(slides["logits"])))
    patient_logits = normalize_f(torch.tensor(np.vstack(patients["logits"])))

    patch_label = torch.tensor(pred["labels"])
    slides_label = torch.tensor(slides["labels"])
    patient_label = torch.tensor(patients["labels"])

    # generate metrics
    def get_all_metrics(logits, label):
        acc = Accuracy(task="multiclass",num_classes=7)
        mca = Accuracy(task="multiclass",num_classes=7, average="macro")
        per_cls = MulticlassAccuracy(num_classes=7, average=None)

        acc_val = acc(logits, label)
        mca_val = mca(logits, label)
        percls_val = per_cls(logits,label)

        return acc_val, mca_val,percls_val

    patch_acc_val, patch_mca_val,patch_percls_val = get_all_metrics(patch_logits, patch_label)
    slide_acc_val, slide_mca_val,slide_percls_val = get_all_metrics(slides_logits, slides_label)
    patient_acc_val, patient_mca_val,patient_percls_val = get_all_metrics(patient_logits, patient_label)

    print(patch_acc_val, patch_mca_val,patch_percls_val)
    print(slide_acc_val, slide_mca_val,slide_percls_val)
    print(patient_acc_val, patient_mca_val,patient_percls_val)
    
    wandb.log({f"eval_knn/patch_acc": get_all_metrics(patch_logits, patch_label)[0]})
    wandb.log({f"eval_knn/slide_acc": get_all_metrics(slides_logits, slides_label)[0]})
    wandb.log({f"eval_knn/patient_acc": get_all_metrics(patient_logits,patient_label)[0]})
    wandb.log({f"eval_knn/patch_mca": get_all_metrics(patch_logits, patch_label)[1]})
    wandb.log({f"eval_knn/slide_mca": get_all_metrics(slides_logits, slides_label)[1]})
    wandb.log({f"eval_knn/patient_mca": get_all_metrics(patient_logits,patient_label)[1]})

    return


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    instance_name = cf["eval"]["ckpt_path"].split("/")[0]
    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, instance_name, "evals",
                            eval_instance_name)

    # generate needed folders, evals will be embedded in experiment folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    for dir_name in [pred_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # if there is a previously generated prediction, also return the
    # prediction filename so we don't have to predict again
    if cf["eval"].get("eval_predictions", None):
        other_eval_instance_name = cf["eval"]["eval_predictions"]
        pred_fname = os.path.join(log_root, exp_name, instance_name, "evals",
                                  other_eval_instance_name, "predictions",
                                  "predictions.pt")
    else:
        pred_fname = None

    return exp_root, pred_dir, partial(copy2, dst=config_dir), pred_fname


def main():
    """Driver script for evaluation pipeline."""
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config, pred_fname = setup_eval_paths(
        cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # get predictions
    if not cf["eval"].get("eval_predictions", None):
        logging.info("generating predictions")
        predictions = get_embeddings(cf, exp_root)
        torch.save(predictions, os.path.join(pred_dir, "predictions.pt"))
    else:
        logging.info("loading predictions")
        predictions = torch.load(pred_fname)

    # generate specs
    make_specs(predictions)


if __name__ == "__main__":
    main()
