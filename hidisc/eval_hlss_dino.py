"""Evaluation modules and script.

Copyright (c) 2023 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import argparse
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any
import tifffile
import csv
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torchmetrics import AveragePrecision, Accuracy
from torchvision import models as torchvision_models
import torch.distributed as dist
import pytorch_lightning as pl

from datasets.srh_dataset import OpenSRHDataset
from datasets.improc import get_srh_base_aug, get_srh_vit_base_aug,get_srh_base_aug_hidisc
from common import (parse_args, get_exp_name, config_loggers,
                           get_num_worker)

from models import MLP, resnet_backbone, ContrastiveLearningNetwork, HLSSContrastiveLearningNetwork, CLIPTextClassifier,CLIPVisual
import utils
import vision_transformer as vits
from vision_transformer import DINOHead, CLIPTextDINOHead

import warnings
warnings.filterwarnings("ignore")

import wandb

wandb.init(project="HLSS")

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='resnet50', type=str,
        choices=['vit_tiny','RN50','vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--out_dim', default=128, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    # Misc
    parser.add_argument('--data_path', default='/data1/dri/hidisc/hidisc/datasets/opensrh', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/data1/dri/hidisc/hidisc/exps/Exp018/b", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--config", default="", type=str, help="config file")
    return parser

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


def get_embeddings(cf: Dict[str, Any],ckpt:str,
                   exp_root: str,train_loader,val_loader) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""

    if cf["model"]["backbone"] == "RN50":
        student = CLIPVisual(arch=cf["model"]["backbone"])
        embed_dim = 1024
    elif cf["model"]["backbone"] == "resnet50":
        student = resnet_backbone(arch=cf["model"]["backbone"])
        embed_dim = 2048
    
    # student = utils.MultiCropWrapper(student, DINOHead(
    #     embed_dim,
    #     cf["model"]["out_dim"],
    #     nlayers = 1,
    #     final_layer=False
    # ))
    ckpt_path = os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_dir"],ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {}
    for key in ckpt['student']:
        if key.startswith('module.backbone'):
                suffix = key[len('module.backbone.'):]
                state_dict[suffix] = ckpt['student'][key]
    # keys_to_remove = ["head.last_layer.weight_g", "head.last_layer.weight_v"]
    # for key in keys_to_remove:
    #     state_dict.pop(key, None)
    student.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    student.eval()

    train_predictions = {"path": [],"label": [],"embeddings": []}
    val_predictions = {"path": [],"label": [],"embeddings": []}

    for i,batch in enumerate(tqdm(train_loader,total=len(train_loader))):
        # assert len(batch["image"].shape) == 4
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            out = student(images)
        train_predictions["path"].extend(batch["path"])  
        train_predictions["label"].append(labels)
        train_predictions["embeddings"].append(out)
    train_predictions["label"] = torch.cat(train_predictions["label"], dim=0)
    train_predictions["embeddings"] = torch.cat(train_predictions["embeddings"], dim=0)
    train_predictions["path"]  = [item for sublist in train_predictions["path"] for item in sublist]

    for i,batch in enumerate(tqdm(val_loader,total=len(val_loader))):
        # assert len(batch["image"].shape) == 4
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            out = student(images)
        val_predictions["path"].extend(batch["path"])  
        val_predictions["label"].append(labels)
        val_predictions["embeddings"].append(out)
    val_predictions["label"] = torch.cat(val_predictions["label"], dim=0)
    val_predictions["embeddings"] = torch.cat(val_predictions["embeddings"], dim=0)
    val_predictions["path"]  = [item for sublist in val_predictions["path"] for item in sublist]

    del student

    # def process_predictions(predictions):
    #     pred = {}
    #     for k in predictions[0].keys():
    #         if k == "path":
    #             pred[k] = [pk for p in predictions for pk in p[k][0]]
    #         else:
    #             pred[k] = torch.cat([p[k] for p in predictions])
    #     return pred

    # train_predictions = process_predictions(train_predictions)
    # val_predictions = process_predictions(val_predictions)

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


def make_specs(epoch_no, predictions: Dict[str, Union[torch.Tensor, List[str]]]) -> None:
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
                                get_all_metrics(patient_logits,patient_label)))
    
    wandb.log({f"eval_knn/patch_acc": get_all_metrics(patch_logits, patch_label)[0]})
    wandb.log({f"eval_knn/slide_acc": get_all_metrics(slides_logits, slides_label)[0]})
    wandb.log({f"eval_knn/patient_acc": get_all_metrics(patient_logits,patient_label)[0]})
    wandb.log({f"eval_knn/patch_mca": get_all_metrics(patch_logits, patch_label)[1]})
    wandb.log({f"eval_knn/slide_mca": get_all_metrics(slides_logits, slides_label)[1]})
    wandb.log({f"eval_knn/patient_mca": get_all_metrics(patient_logits,patient_label)[1]})

    csv_filename = "exp18b_eval.csv"
    
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write a new row with the provided values
        writer.writerow([epoch_no,get_all_metrics(patch_logits, patch_label)[0], get_all_metrics(slides_logits, slides_label)[0], get_all_metrics(patient_logits,patient_label)[0]])



    return


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    instance_name = cf["eval"]["ckpt_dir"].split("/")[0]
    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, "evals")

    # generate needed folders, evals will be embedded in experiment folders
    pred_dir = os.path.join(exp_root, 'predictions')
    print(f'pred dir {pred_dir}')
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

    #create dataset
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

    # get predictions
    if not cf["eval"].get("eval_predictions", None):
        logging.info("generating predictions")
        ckpt_list_old = os.listdir(os.path.join(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                             cf["eval"]["ckpt_dir"]))
        ckpt_list = ['checkpoint0040.pth','checkpoint0080.pth','checkpoint0120.pth',
                     'checkpoint0160.pth','checkpoint0200.pth','checkpoint0240.pth',
                     'checkpoint0280.pth']
        # ckpt_list = ['checkpoint0280.pth']
        
        csv_filename = "exp18b_eval.csv"
        try:
            df = pd.read_csv(csv_filename)
            
        except FileNotFoundError:
            columns = ['epoch','patch', 'slide', 'patient']
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_filename, index=False)

        for epoch,ckpt in tqdm(enumerate(ckpt_list)):
            epoch_str = ckpt_list[epoch].split("checkpoint")[1].split(".")[0]
            epoch_no = int(epoch_str)
            print(f'ckpt {ckpt}')
            predictions = get_embeddings(cf,ckpt, exp_root,train_loader,val_loader)
            predpath = (ckpt.split(".")[0])

            torch.save(predictions, os.path.join(pred_dir, f"predictions_{predpath}.pt"))
            make_specs(epoch_no,predictions)

            
    else:
        logging.info("loading predictions")
        predictions = torch.load(pred_fname)


if __name__ == "__main__":
    main()

