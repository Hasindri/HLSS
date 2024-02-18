"""Evaluation modules and script.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import hydra

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
from torchvision.transforms import Compose
from helpers import setup_seed

from torchmetrics import AveragePrecision, Accuracy

from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug
# from helpers import (parse_args, get_exp_name, config_loggers,
#                                  get_num_worker)
from datasets.loaders import get_num_worker
from main import HiDiscModel
log = logging.getLogger(__name__)


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

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        # num_workers=get_num_worker(),
        persistent_workers=False)

    val_dset = OpenSRHDataset(data_root=cf["data"]["db_root"],
                              studies="val",
                              transform=Compose(aug_func()),
                              balance_patch_per_class=False)

    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=cf["eval"]["predict_batch_size"],
        drop_last=False,
        pin_memory=True,
        # num_workers=get_num_worker(),
        persistent_workers=False)

    # load model from checkpoint
    ckpt_path = cf["eval"]["ckpt_path"]
    model = HiDiscModel(cf)
    msg = model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    log.info(f"Loaded model from {ckpt_path} with message {msg}")
    model.to("cuda")
    model.eval()

    train_predictions = []
    for i, batch in enumerate(train_loader):

        images = batch["image"].to("cuda")
        labels = batch["label"].to("cuda")
        with torch.no_grad():
            outputs = model.get_features(images)

        d = {"embeddings": outputs, "label": labels, "path": batch["path"]}
        log.info(f"BATCH {i} {d['embeddings'].shape} {d['label'].shape}")

        train_predictions.append(d)

    val_predictions = []

    for i, batch in enumerate(val_loader):

        images = batch["image"].to("cuda")
        labels = batch["label"].to("cuda")
        with torch.no_grad():
            outputs = model.get_features(images)

        d = {"embeddings": outputs, "label": labels, "path": batch["path"]}

        log.info(f"BATCH {i} {d['embeddings'].shape} {d['label'].shape}")
        val_predictions.append(d)

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

    log.info(train_predictions["embeddings"].shape)
    log.info(train_predictions["label"].shape)

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
        map = AveragePrecision(num_classes=7, task="binary")
        acc = Accuracy(num_classes=7, task="binary")
        t2 = Accuracy(num_classes=7, top_k=2, task="binary")
        t3 = Accuracy(num_classes=7, top_k=3, task="binary")
        mca = Accuracy(num_classes=7, average="macro", task="binary")

        logits = logits.squeeze()  # Remove extra dimensions from logits
        acc_val = acc(logits, label)
        t2_val = t2(logits, label)
        t3_val = t3(logits, label)
        mca_val = mca(logits, label)
        map_val = map(logits, label)

        return torch.stack((acc_val, t2_val, t3_val, mca_val, map_val))

    all_metrics = torch.vstack((get_all_metrics(patch_logits, patch_label),
                                get_all_metrics(slides_logits, slides_label),
                                get_all_metrics(patient_logits,
                                                patient_label)))
    all_metrics = pd.DataFrame(all_metrics,
                               columns=["acc", "t2", "t3", "mca", "map"],
                               index=["patch", "slide", "patient"])

    # generate confusion matrices
    patch_conf = confusion_matrix(y_true=patch_label,
                                  y_pred=patch_logits.argmax(dim=1))

    slide_conf = confusion_matrix(y_true=slides_label,
                                  y_pred=slides_logits.argmax(dim=1))

    patient_conf = confusion_matrix(y_true=patient_label,
                                    y_pred=patient_logits.argmax(dim=1))

    log.info("\nmetrics")
    log.info(all_metrics)
    log.info("\npatch confusion matrix")
    log.info(patch_conf)
    log.info("\nslide confusion matrix")
    log.info(slide_conf)
    log.info("\npatient confusion matrix")
    log.info(patient_conf)

    return



@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cf):

    setup_seed(cf["seed"])

    log.info("Info level message")
    # log.debug("Debug level message")

    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {get_original_cwd()}")

    prediction_path = os.path.join(cf['eval']['save_predictions_path'], "predictions.pt")

    if os.path.exists(prediction_path):
        log.info("loading predictions")
        predictions = torch.load(prediction_path)
    else:
        log.info("generating predictions")
        predictions = get_embeddings(cf, None)
        torch.save(predictions, prediction_path)

    # generate specs
    make_specs(predictions)


if __name__ == "__main__":
    main()