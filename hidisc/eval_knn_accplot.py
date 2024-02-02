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
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
from torchvision.transforms import Compose

import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy

from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug, get_srh_base_aug_hidisc
from common import (parse_args, get_exp_name, config_loggers,
                           get_num_worker)
from train_hidisc import HiDiscSystem

import wandb

wandb.init(project="HLSS")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def get_embeddings(cf: Dict[str, Any], ckpt:str,
                   exp_root: str,train_loader,val_loader) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    # get model
    # if cf["model"]["backbone"] == "resnet50":
    #     aug_func = get_srh_base_aug_hidisc
    # elif cf["model"]["backbone"] == "RN50":
    #     aug_func = get_srh_base_aug
    # elif cf["model"]["backbone"] == "vit":
    #     aug_func = get_srh_vit_base_aug
    # else:
    #     raise NotImplementedError()

    # # get dataset / loader
    # train_dset = OpenSRHDataset(data_root=cf["data"]["db_root"],
    #                             studies="train",
    #                             transform=Compose(aug_func()),
    #                             balance_patch_per_class=False)
    # train_dset.reset_index()


    # train_loader = torch.utils.data.DataLoader(
    #     train_dset,
    #     batch_size=cf["eval"]["predict_batch_size"],
    #     drop_last=False,
    #     pin_memory=True,
    #     num_workers=get_num_worker(),
    #     persistent_workers=True)

    # val_dset = OpenSRHDataset(data_root=cf["data"]["db_root"],
    #                           studies="val",
    #                           transform=Compose(aug_func()),
    #                           balance_patch_per_class=False)
    # val_dset.reset_index()


    # val_loader = torch.utils.data.DataLoader(
    #     val_dset,
    #     batch_size=cf["eval"]["predict_batch_size"],
    #     drop_last=False,
    #     pin_memory=True,
    #     num_workers=get_num_worker(),
    #     persistent_workers=True)


    # logging.info(f"num devices: {torch.cuda.device_count()}")
    # logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    # load lightning checkpoint
    ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_dir"],ckpt)
    
    model = HiDiscSystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0,
                                              max_epochs=-1,
                                              nc=0,freeze_mlp=True)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=-1,
                         default_root_dir=exp_root,
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
    # print(f' val predictions {val_predictions}')

    train_embs = torch.nn.functional.normalize(train_predictions["embeddings"],
                                               p=2,
                                               dim=1).T
    val_embs = torch.nn.functional.normalize(val_predictions["embeddings"],
                                             p=2,
                                             dim=1)
    # print(f'val_embs {val_embs.shape}')

    # print(f'train_predictions embeddings {train_predictions["embeddings"].shape}')
    # print(f'train_predictions labels {train_predictions["label"].shape}')

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
    del model
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
        # map = AveragePrecision(num_classes=7,task="multiclass")
        acc = Accuracy(task="multiclass",num_classes=7)
        # t2 = Accuracy(task="multiclass",num_classes=7, top_k=2)
        # t3 = Accuracy(task="multiclass",num_classes=7, top_k=3)
        mca = Accuracy(task="multiclass",num_classes=7, average="macro")

        acc_val = acc(logits, label)
        # t2_val = t2(logits, label)
        # t3_val = t3(logits, label)
        mca_val = mca(logits, label)
        # map_val = map(logits, label)

        # return torch.stack((acc_val, t2_val, t3_val, mca_val, map_val))
        return torch.stack((acc_val, mca_val))

    all_metrics = torch.vstack((get_all_metrics(patch_logits, patch_label),
                                get_all_metrics(slides_logits, slides_label),
                                get_all_metrics(patient_logits,
                                                patient_label)))
    

    wandb.log({f"eval_knn/patch_acc": get_all_metrics(patch_logits, patch_label)[0]})
    wandb.log({f"eval_knn/slide_acc": get_all_metrics(slides_logits, slides_label)[0]})
    wandb.log({f"eval_knn/patient_acc": get_all_metrics(patient_logits,patient_label)[0]})
    wandb.log({f"eval_knn/patch_mca": get_all_metrics(patch_logits, patch_label)[1]})
    wandb.log({f"eval_knn/slide_mca": get_all_metrics(slides_logits, slides_label)[1]})
    wandb.log({f"eval_knn/patient_mca": get_all_metrics(patient_logits,patient_label)[1]})

    # all_metrics = pd.DataFrame(all_metrics,
    #                            columns=["acc", "t2", "t3", "mca", "map"],
    #                            index=["patch", "slide", "patient"])

    # # generate confusion matrices
    # patch_conf = confusion_matrix(y_true=patch_label,
    #                               y_pred=patch_logits.argmax(dim=1))

    # slide_conf = confusion_matrix(y_true=slides_label,
    #                               y_pred=slides_logits.argmax(dim=1))

    # patient_conf = confusion_matrix(y_true=patient_label,
    #                                 y_pred=patient_logits.argmax(dim=1))

    # print("\nmetrics")
    # print(all_metrics)
    # print("\npatch confusion matrix")
    # print(patch_conf)
    # print("\nslide confusion matrix")
    # print(slide_conf)
    # print("\npatient confusion matrix")
    # print(patient_conf)

    return


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    instance_name = cf["eval"]["ckpt_dir"].split("/")[0]
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

    #dataset
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

    # get predictions
    if not cf["eval"].get("eval_predictions", None):
        logging.info("generating predictions")
        ckpt_list = os.listdir(os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_dir"]))
        # print(f'ckpt list {len(ckpt_list)}')
        ckpt_list_800 = []
        epoch_list_800 = []

        for ckpt_string in ckpt_list:
            epoch_str = ckpt_string.split("epoch")[1].split(".")[0]
            epoch = int(epoch_str)
            
            if (epoch + 1) % 800 == 0:
                epoch_list_800.append(epoch)

        epoch_list_800 = [epoch for epoch in epoch_list_800 if epoch >= 4800]
        epoch_list_800.sort()
        print(f'epoch list {epoch_list_800}')

        for i in epoch_list_800:
            ckpt = f"ckpt-epoch{i}.ckpt"
            ckpt_list_800.append(ckpt)

        print(f'ckpt list {ckpt_list_800}')

        for ckpt in tqdm(ckpt_list_800):
            print(f'ckpt {ckpt}')
            predictions = get_embeddings(cf,ckpt, exp_root,train_loader,val_loader)
            predpath = (ckpt.split(".")[0]).split("-")[1]
            # print(f'predpath {predpath}')
            # epoch = int(predpath[5:])

        
            torch.save(predictions, os.path.join(pred_dir, f"predictions_{predpath}.pt"))
            make_specs(predictions)
   
    else:
        logging.info("loading predictions")
        predictions = torch.load(pred_fname)


if __name__ == "__main__":
    main()

