"""PyTorch datasets designed to work with OpenSRH.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import json
import logging
from collections import Counter
from typing import Optional, List, Union, TypedDict, Tuple
import random

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import Compose
from torchvision.transforms import (
    Normalize, RandomApply, Compose, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, RandAugment, RandomErasing, RandomAutocontrast, Grayscale,
    RandomSolarize, ColorJitter, RandomAdjustSharpness, GaussianBlur,
    RandomAffine, RandomResizedCrop)

from datasets.improc import process_read_im, get_srh_base_aug, get_tcga_base_aug, read_h5_patches,read_one_patch,read_400_h5_patches


class PatchData(TypedDict):
    image: Optional[torch.Tensor]
    label: Optional[torch.Tensor]
    path: Optional[List[str]]


class OpenSRHDataset(Dataset):
    """OpenSRH classification dataset - used for evaluation"""

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 transform: callable = Compose(get_srh_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_patch_per_class: bool = False,
                 check_images_exist: bool = False) -> None:
        """Inits the OpenSRH dataset.
        
        Populate each attribute and walk through slides to look for patches.

        Args:
            data_root: root OpenSRH directory
            studies: either a string in {"train", "val"} for the default
                train/val dataset split, or a list of strings representing
                patient IDs
            transform: a callable object for image transformation
            target_transform: a callable object for label transformation
            balance_patch_per_class: balance the patches in each class
            check_images_exist: a flag representing whether to check every
                image file exists in data_root. Turn this on for debugging,
                turn it off for speed.
        """

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.check_images_exist_ = check_images_exist
        self.get_all_meta()
        self.get_study_list(studies)

        # Walk through each study
        self.instances_ = []
        for p in tqdm(self.studies_):
            self.instances_.extend(self.get_study_instances(p))
        print(f'total {studies} patch count {len(self.instances_)}')

        if balance_patch_per_class:
            self.replicate_balance_instances()
        self.get_weights()

    def get_all_meta(self):
        """Read in all metadata files."""

        try:
            with open(os.path.join(self.data_root_,
                                   "meta/updated2_opensrh.json")) as fd:
                self.metadata_ = json.load(fd)
        except Exception as e:
            logging.critical("Failed to locate dataset.")
            raise e

        logging.info(f"Locate OpenSRH dataset at {self.data_root_}")
        return

    def get_study_list(self, studies):
        """Get a list of studies from default split or list of IDs."""

        if isinstance(studies, str):
            try:
                with open(
                        os.path.join(self.data_root_,
                                     "meta/updated2_train_val_split.json")) as fd:
                    train_val_split = json.load(fd)
            except Exception as e:
                logging.critical("Failed to locate preset train/val split.")
                raise e

            if studies == "train":
                self.studies_ = train_val_split["train"]
                print(f'train patient count {len(self.studies_)}')
            elif studies in ["valid", "val"]:
                self.studies_ = train_val_split["val"]
                print(f'val patient count {len(self.studies_)}')
            else:
                return ValueError(
                    "studies split must be one of [\"train\", \"val\"]")
        elif isinstance(studies, List):
            self.studies_ = studies
        else:
            raise ValueError("studies must be a string representing " +
                             "train/val split or a list of study numbers")
        return

    def get_study_instances(self, patient: str):
        """Get all instances from one study."""

        slide_instances = []
        # logging.debug(patient)
        if self.check_images_exist_:
            tiff_file_exist = lambda im_p: (os.path.exists(im_p) and
                                            is_image_file(im_p))
        else:
            tiff_file_exist = lambda _: True

        def check_add_patches(patches: List[str]):
            for p in patches:
                im_p = os.path.join(self.data_root_, p)
                if tiff_file_exist(im_p):
                    slide_instances.append(
                        (im_p, self.metadata_[patient]["class"]))
                else:
                    logging.warning(f"Bad patch: unable to locate {im_p}")

        for s in self.metadata_[patient]["slides"]:
            if self.metadata_[patient]["class"] == "normal":
                check_add_patches(
                    self.metadata_[patient]["slides"][s]["normal_patches"])
            else:
                check_add_patches(
                    self.metadata_[patient]["slides"][s]["tumor_patches"])
        # logging.debug(f"patient {patient} patches {len(slide_instances)}")
        return slide_instances

    def process_classes(self):
        """Look for all the labels in the dataset.

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [i[1] for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))
        return

    def get_weights(self):
        """Count number of instances for each class, and computes weights."""

        # Get classes
        self.process_classes()
        all_labels = [self.class_to_idx_[i[1]] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        # logging.debug("Weights: {}".format(self.weights_))
        return self.weights_

    def replicate_balance_instances(self):
        """resample the instances list to balance each class."""
        all_labels = [i[1] for i in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [i for i in self.instances_ if i[1] == l]
            random.shuffle(instances_l)
            instances_l = instances_l * (val_sample // len(instances_l) + 1)
            all_instances_.extend(sorted(instances_l[:val_sample]))

        self.instances_ = all_instances_
        return

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.instances_)

    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve a patch specified by idx"""

        # print(f'idx {idx}')
        # print(f'all patches {len(self.instances_)}')
        imp, target = self.instances_[idx]
        target = self.class_to_idx_[target]

        # Read image
        logging.debug("imp: {}".format(imp))
        # print(f'imp {imp}')
        im: torch.Tensor = process_read_im(imp)
        # print(f'im {im.shape}')

        # Perform transformations
        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}

    def reset_index(self):
        """Reset the index of the dataset instances."""
        self.instances_ = list(pd.DataFrame(self.instances_).reset_index(drop=True).to_records(index=False))


class HiDiscDataset(Dataset):
    """HiDisc dataset for OpenSRH"""

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 num_slide_samples: int = 2,
                 num_patch_samples: int = 2,
                 num_transforms: int = 2,
                 transform: callable = Compose(get_srh_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_study_per_class: bool = False,
                 check_images_exist: bool = False) -> None:
        """Initializes the HiDisc Dataset for OpenSRH

        Populate each attribute and walk through slides to look for patches.

        Args:
            data_root: root OpenSRH directory
            studies: either a string in {"train", "val"} for the default
                train/val dataset split, or a list of strings representing
                patient IDs
            num_slide_samples: number of slides to sample in each patient
            num_patch_samples: number of patches to sample in each slide
            num_transforms: number of views (augmentations) for each patch
            transform: a callable object for image transformation
            target_transform: a callable object for label transformation
            balance_study_per_class: balance the patients in each class
            check_images_exist: a flag representing whether to check every
                image file exists in data_root. Turn this on for debugging,
                turn it off for speed.
        """

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.check_images_exist_ = check_images_exist
        self.num_slide_samples_ = num_slide_samples
        self.num_patch_samples_ = num_patch_samples
        self.num_transforms_ = num_transforms
        self.get_all_meta()
        self.get_study_list(studies)

        # Walk through each study
        self.instances_ = []
        for p in tqdm(self.studies_):
            self.instances_.append(
                (self.get_study_instances(p), self.metadata_[p]["class"]))
        
        #self.instances_[i] is a 2 element tuple (list of paths to all patches of patent i, class of patient i)

        # print(f'Patient {self.studies_[0]} noof patches {len(self.instances_[0][0][0])} class {self.instances_[0][1]}')
        if balance_study_per_class:
            self.replicate_balance_instances()
        self.get_weights()

    def get_all_meta(self):
        """Read in all metadata files."""

        try:
            with open(os.path.join(self.data_root_,
                                   "meta/updated2_opensrh.json")) as fd:
                self.metadata_ = json.load(fd)
        except Exception as e:
            logging.critical("Failed to locate dataset.")
            raise e

        logging.info(f"Locate OpenSRH dataset at {self.data_root_}")
        return

    def get_study_list(self, studies):
        """Get a list of studies from default split or list of IDs."""

        if isinstance(studies, str):
            try:
                with open(
                        os.path.join(self.data_root_,
                                     "meta/updated2_train_val_split.json")) as fd:
                    train_val_split = json.load(fd)
            except Exception as e:
                logging.critical("Failed to locate preset train/val split.")
                raise e

            if studies == "train":
                self.studies_ = train_val_split["train"]
                # print(f'studies list {self.studies_}')
            elif studies in ["valid", "val"]:
                self.studies_ = train_val_split["val"]
                # print(f'studies list {self.studies_}')
            else:
                return ValueError(
                    "studies split must be one of [\"train\", \"val\"]")
        elif isinstance(studies, List):
            self.studies_ = studies
        else:
            raise ValueError("studies must be a string representing " +
                             "train/val split or a list of study numbers")
        return

    def get_study_instances(self, patient: str) -> List[List[str]]:
        """Get all instances from one study."""

        one_patient_instance: List[List[str]] = []  # List of slides
        # logging.debug(f"patient {patient}")

        if self.check_images_exist_:
            tiff_file_exist = lambda im_p: (os.path.exists(im_p) and
                                            is_image_file(im_p))
        else:
            tiff_file_exist = lambda _: True

        def make_slide_instance(patches: List[str]) -> List[str]:
            good_patches: List[str] = []  # List of patches
            for p in patches:
                im_p = os.path.join(self.data_root_, p)
                if tiff_file_exist(im_p):
                    good_patches.append(im_p)
                else:
                    logging.warning(f"Bad patch: unable to locate {im_p}")
            return good_patches

        for s in self.metadata_[patient]["slides"]:
            if self.metadata_[patient]["class"] == "normal":
                si = make_slide_instance(
                    self.metadata_[patient]["slides"][s]["normal_patches"])
            else:
                si = make_slide_instance(
                    self.metadata_[patient]["slides"][s]["tumor_patches"])
            # logging.debug(f"patient {patient}\tslide {s} \tpatches {len(si)}")
            if len(si):
                one_patient_instance.append(si)

        # logging.debug(
        #     f"patient {patient} total slides {len(one_patient_instance)}")
    
        return one_patient_instance

    def process_classes(self):
        """Look for all the labels in the dataset.

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [i[1] for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))
        return

    def get_weights(self):
        """Count number of instances for each class, and computes weights."""

        # Get classes
        self.process_classes()
        all_labels = [self.class_to_idx_[i[1]] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        # logging.debug("Weights: {}".format(self.weights_))
        return self.weights_

    def replicate_balance_instances(self):
        """resample the instances list to balance each class."""

        all_labels = [i[1] for i in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [i for i in self.instances_ if i[1] == l]
            random.shuffle(instances_l)
            instances_l = instances_l * (val_sample // len(instances_l) + 1)
            all_instances_.extend(sorted(instances_l[:val_sample]))

        self.instances_ = all_instances_
        return

    def read_images_slide(self, inst: List[Tuple]):
        """Read in a list of patches, different patches and transformations"""

        im_id = np.random.permutation(np.arange(len(inst)))
        # print(f'inst {len(inst)}')
        # print(f'im_id {im_id}')
        images = []
        imps_take = []

        # print(f'transforms {self.num_transforms_}')

        idx = 0
        while len(images) < self.num_patch_samples_:
            curr_inst = inst[im_id[idx % len(im_id)]]
            # print(f'curr_inst {curr_inst}')
            # breakpoint()
            try:
                images.append(process_read_im(curr_inst))
                imps_take.append(curr_inst)
                idx += 1
            except:
                logging.error("bad_file - {}".format(curr_inst))
                idx += 1
                # pass
                

        # print(f'wanted noof patches {self.num_patch_samples_}')

        # while len(images) < self.num_patch_samples_:
        #     # print(f'index of curr_inst {im_id[idx % len(im_id)]}')
        #     curr_inst = inst[im_id[idx % len(im_id)]]
        #     print(f'curr_inst {curr_inst}')
        #     try:
        #         with process_read_im(curr_inst) as im:  # Use 'with' statement
        #             images.append(im)
        #             print(f'images {images}')
        #         imps_take.append(curr_inst)
        #         idx += 1
                
        #         # print(f'imps_take {imps_take}')
        #     except Exception as e:
        #         # logging.error("bad_file - {}".format(curr_inst))
        #         # logging.warning(f"Error reading image: {e}")
        #         pass
        #     finally:
        #         if 'im' in locals() and hasattr(im, 'close'):
        #             im.close()

        assert self.transform_ is not None
        # print(f'srh patch0 {images[0].shape}')
        xformed_im = torch.stack([
            torch.stack(
                [self.transform_(im) for _ in range(self.num_transforms_)])
            for im in images
        ])
        # print(f'xformed_im {len(xformed_im)}')
        # print(f'imps_take {len(imps_take)}')
        return xformed_im, imps_take

    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve patches from patient as specified by idx"""

        patient, target = self.instances_[idx]
        # print(f'patient {patient}')
        # print(f'target {target}')

        num_slides = len(patient)
        # print(f'num_slides {num_slides}')
        slide_idx = np.arange(num_slides)
        # print(f'slide idx {slide_idx}')
        np.random.shuffle(slide_idx)
        num_repeat = self.num_slide_samples_ // len(patient) + 1
        # print(f'num repeat {num_repeat}')
        slide_idx = np.tile(slide_idx, num_repeat)[:self.num_slide_samples_]
        # print(f'slide idx {slide_idx}')

        images = [self.read_images_slide(patient[i]) for i in slide_idx]
        #images is a list of tuples, first item of each tuple contains all 
        #transformations of all patches in that slide as tensors, second 
        #element of tuple is list of considered patch paths of that tuple/slide
        # print(f'images {len(images)}')
        # print(f'images[0] {images[0][0]}')
        im = torch.stack([i[0] for i in images])
        # print(f'image {im.shape}')
        imp = [i[1] for i in images]
        # print(f'imp {imp}')

        target = self.class_to_idx_[target]
        # print(f'target {target}')
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}

    def __len__(self):
        return len(self.instances_)

    # def __getitem__(self, idx: int) -> Union[PatchData, None]:
    #     patient, target = self.instances_[idx]
    
    #     # Check if the patient ID exists in the updated metadata file
    #     if patient not in self.metadata_:
    #         print(f'patient ID {patient} is not in metadata')
    #         return None
        
    #     num_slides = len(self.metadata_[patient]["slides"])
    #     slide_idx = np.arange(num_slides)
    #     np.random.shuffle(slide_idx)
    #     num_repeat = self.num_slide_samples_ // len(patient) + 1
    #     slide_idx = np.tile(slide_idx, num_repeat)[:self.num_slide_samples_]

    #     images = [self.read_images_slide(patient[i]) for i in slide_idx]
    #     im = torch.stack([i[0] for i in images])
    #     imp = [i[1] for i in images]

    #     target = self.class_to_idx_[target]
    #     if self.target_transform_ is not None:
    #         target = self.target_transform_(target)

    #     return {"image": im, "label": target, "path": [imp]}

class HiDiscDataset_TCGA(Dataset):
    """HiDisc dataset for TCGA"""

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 num_slide_samples: int = 2,
                 num_patch_samples: int = 2,
                 num_transforms: int = 2,
                 transform: callable = Compose(get_tcga_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_study_per_class: bool = False,
                 check_images_exist: bool = False) -> None:
        """Initializes the HiDisc Dataset for TCGA

        Populate each attribute and walk through slides to look for patches.

        Args:
            data_root: root TCGA directory
            studies: either a string in {"train", "val"} for the default
                train/val dataset split, or a list of strings representing
                patient IDs
            num_slide_samples: number of slides to sample in each patient
            num_patch_samples: number of patches to sample in each slide
            num_transforms: number of views (augmentations) for each patch
            transform: a callable object for image transformation
            target_transform: a callable object for label transformation
            balance_study_per_class: balance the patients in each class
            check_images_exist: a flag representing whether to check every
                image file exists in data_root. Turn this on for debugging,
                turn it off for speed.
        """

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.check_images_exist_ = check_images_exist
        self.num_slide_samples_ = num_slide_samples
        self.num_patch_samples_ = num_patch_samples
        self.num_transforms_ = num_transforms
        self.get_all_meta()
        self.get_study_list(studies)

        # Walk through each study
        self.instances_ = []
        # print(f'{studies} split : {self.studies_}')
        for p in tqdm(self.studies_):
            self.instances_.append(
                (self.get_study_instances(p), self.metadata_[p]["class"]))
        
        #self.instances_[i] is a 2 element tuple (list of paths to all patches of patent i, class of patient i)

        #print(f'Patient {self.studies_[0]} noof slides {len(self.instances_[0][0])} class {self.instances_[0][1]}')
        if balance_study_per_class:
            self.replicate_balance_instances()
        self.get_weights()

    def get_all_meta(self):
        """Read in all metadata files."""

        try:
            with open(os.path.join(self.data_root_,
                                   "meta/tcga_lgggbm.json")) as fd:
                self.metadata_ = json.load(fd)
        except Exception as e:
            logging.critical("Failed to locate dataset.")
            raise e

        logging.info(f"Locate TCGA dataset at {self.data_root_}")
        return

    def get_study_list(self, studies):
        """Get a list of studies from default split or list of IDs."""

        if isinstance(studies, str):
            try:
                with open(
                        os.path.join(self.data_root_,
                                     "meta/tcga_lgggbm_trainval_split_cleaned.json")) as fd:
                    train_val_split = json.load(fd)
            except Exception as e:
                logging.critical("Failed to locate preset train/val split.")
                raise e

            if studies == "train":
                self.studies_ = train_val_split["train"]
                # print(f'studies list {self.studies_}')
            elif studies in ["valid", "val"]:
                self.studies_ = train_val_split["val"]
                #print(f'studies list {self.studies_}')
            else:
                return ValueError(
                    "studies split must be one of [\"train\", \"val\"]")
        elif isinstance(studies, List):
            self.studies_ = studies
        else:
            raise ValueError("studies must be a string representing " +
                             "train/val split or a list of study numbers")
        return

    def get_study_instances(self, patient: str) -> List[List[str]]:
        """Get all instances from one study."""

        one_patient_instance: List[List[str]] = []  # List of slides
        # logging.debug(f"patient {patient}")

        # if self.check_images_exist_:
        #     tiff_file_exist = lambda im_p: (os.path.exists(im_p) and
        #                                     is_image_file(im_p))
        # else:
        #     tiff_file_exist = lambda _: True

        def make_slide_instance(patches: List[str]) -> List[str]:
            good_patches: List[str] = []  # List of patches
            for p in patches:
                im_p = os.path.join(self.data_root_, p)
                if tiff_file_exist(im_p):
                    good_patches.append(im_p)
                else:
                    logging.warning(f"Bad patch: unable to locate {im_p}")
            return good_patches

        for s in self.metadata_[patient]["slides"]:
            si = self.metadata_[patient]["slides"][s]["patch_path"]
            logging.debug(f"patient {patient}\tslide {s} \tpatchpth {si}")
            if si:
                one_patient_instance.append(si)

        # logging.debug(
        #     f"patient {patient} total slides {len(one_patient_instance)}")

        #print(f"patient {patient} total slides {len(one_patient_instance)}")
        #one_patient_instance is a [] with all patch_paths of that patient
    
        return one_patient_instance

    def process_classes(self):
        """Look for all the labels in the dataset.

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [i[1] for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))
        return

    def get_weights(self):
        """Count number of instances for each class, and computes weights."""

        # Get classes
        self.process_classes()
        all_labels = [self.class_to_idx_[i[1]] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        # logging.debug("Weights: {}".format(self.weights_))
        return self.weights_

    def replicate_balance_instances(self):
        """resample the instances list to balance each class."""

        all_labels = [i[1] for i in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [i for i in self.instances_ if i[1] == l]
            random.shuffle(instances_l)
            instances_l = instances_l * (val_sample // len(instances_l) + 1)
            all_instances_.extend(sorted(instances_l[:val_sample]))

        self.instances_ = all_instances_
        return

    def read_images_slide(self, inst: List[Tuple]):
        """Read in a list of patches, different patches and transformations"""
        #inst is the patch_path of one slide

        # A tensor containing image patches of shape (n, 300, 300, 3)
        # try:
        #     inst_h5 = read_h5_patches(os.path.join(self.data_root_,inst[0]))
        #     #print(f'slide path {inst[0]}')
        #     #print(f'inst_h5 {inst_h5.shape}')
        # except:
        #     logging.error("bad_file - {}".format(inst[0]))
            


        # im_id = np.random.permutation(np.arange(inst_h5.shape[0]))


        images = []
        imps_take = []

        # print(f'transforms {self.num_transforms_}')

        idx = 0
        while len(images) < self.num_patch_samples_:

            # curr_idx = idx % len(im_id)
        
            # curr_inst = inst_h5[im_id[curr_idx]]
            curr_inst, sel_idx = read_one_patch(os.path.join(self.data_root_,inst[0]),idx)
            curr_inst = curr_inst.permute(2, 0, 1)
            # print(f'curr idx {im_id[curr_idx]}')
            # print(f'curr_inst {curr_inst.shape}')
   
            images.append(curr_inst)
            imps_take.append(sel_idx)
            idx += 1


        assert self.transform_ is not None

        xformed_im = torch.stack([
            torch.stack(
                [self.transform_(im) for _ in range(self.num_transforms_)])
            for im in images
        ])

        return xformed_im, imps_take

    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve patches from patient as specified by idx"""
        # print(f'idx {idx}')
        patient, target = self.instances_[idx]
        # print(f'patient {patient}')
        #print(f'target {target}')

        num_slides = len(patient)
        # print(f'num_slides {num_slides}')
        slide_idx = np.arange(num_slides)
        # print(f'slide idx {slide_idx}')
        np.random.shuffle(slide_idx)
        num_repeat = self.num_slide_samples_ // len(patient) + 1
        #print(f'num repeat {num_repeat}')
        slide_idx = np.tile(slide_idx, num_repeat)[:self.num_slide_samples_]
        # print(f'slide idx {slide_idx}')

        images = [self.read_images_slide(patient[i]) for i in slide_idx]
        #images is a list of tuples, first item of each tuple contains all 
        #transformations of all patches in that slide as tensors, second 
        #element of tuple is list of considered patch paths of that tuple/slide
        # print(f'images {len(images)}')
        # print(f'images[0] {images[0][0][0].shape}')
        im = torch.stack([i[0] for i in images])
        #print(f'image {im.shape}')
        imp = [i[1] for i in images]
        #print(f'imp {imp}')

        target = self.class_to_idx_[target]
        #print(f'target {target}')
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}

    def __len__(self):
        return len(self.instances_)


    def reset_index(self):
        """Reset the index of the dataset instances."""
        self.instances_ = list(pd.DataFrame(self.instances_).reset_index(drop=True).to_records(index=False))


class TCGADataset(Dataset):
    """ dataset for TCGA"""

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 transform: callable = Compose(get_tcga_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_slide_per_class: bool = False,
                 check_images_exist: bool = False) -> None:
        """Initializes the Dataset for TCGA

        Populate each attribute and walk through slides to look for patches.

        Args:
            data_root: root TCGA directory
            studies: either a string in {"train", "val"} for the default
                train/val dataset split, or a list of strings representing
                patient IDs
            transform: a callable object for image transformation
            target_transform: a callable object for label transformation
            balance_study_per_class: balance the patients in each class
            check_images_exist: a flag representing whether to check every
                image file exists in data_root. Turn this on for debugging,
                turn it off for speed.
        """

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.check_images_exist_ = check_images_exist
        self.get_all_meta()
        self.get_study_list(studies)

        # Walk through each study
        self.instances_ = []
        # print(f'{studies} split : {self.studies_}')
        for p in tqdm(self.studies_):
            self.instances_.extend(
                (self.get_study_instances(p)))
        
        #self.instances_[i] is a 2 element tuple (list of paths to all patches of patent i, class of patient i)

        #print(f'Patient {self.studies_[0]} noof slides {len(self.instances_[0][0])} class {self.instances_[0][1]}')
        if balance_slide_per_class:
            self.replicate_balance_instances()
        self.get_weights()

    def get_all_meta(self):
        """Read in all metadata files."""

        try:
            with open(os.path.join(self.data_root_,
                                   "meta/tcga_lgggbm.json")) as fd:
                self.metadata_ = json.load(fd)
        except Exception as e:
            logging.critical("Failed to locate dataset.")
            raise e

        logging.info(f"Locate TCGA dataset at {self.data_root_}")
        return

    def get_study_list(self, studies):
        """Get a list of studies from default split or list of IDs."""

        if isinstance(studies, str):
            try:
                with open(
                        os.path.join(self.data_root_,
                                     "meta/tcga_lgggbm_trainval_split_cleaned.json")) as fd:
                    train_val_split = json.load(fd)
            except Exception as e:
                logging.critical("Failed to locate preset train/val split.")
                raise e

            if studies == "train":
                self.studies_ = train_val_split["train"]
                # print(f'studies list {self.studies_}')
            elif studies in ["valid", "val"]:
                self.studies_ = train_val_split["val"]
                # print(f'studies list {self.studies_}')
            else:
                return ValueError(
                    "studies split must be one of [\"train\", \"val\"]")
        elif isinstance(studies, List):
            self.studies_ = studies
        else:
            raise ValueError("studies must be a string representing " +
                             "train/val split or a list of study numbers")
        return

    def get_study_instances(self, patient: str) -> List[List[str]]:
        """Get 400 patch instances per slide from one study/patient."""

        one_patient_instance = []  # list of 4 item tuples where each tuple represent one patch 
        #with (slide_path, patch_idx, patch_tensor, class)


        for s in self.metadata_[patient]["slides"]:
            si = self.metadata_[patient]["slides"][s]["patch_path"][0]
            # logging.debug(f"patient {patient}\tslide {s} \tpatchpth {si}")
            if si:
                
                stacked_tensor,idx_list = read_400_h5_patches(os.path.join(self.data_root_,si))
                
                for patch_idx in range(len(idx_list)):
                    one_patient_instance.append((si,idx_list[patch_idx],stacked_tensor[patch_idx],self.metadata_[patient]["class"]))

    
        return one_patient_instance

    def process_classes(self):
        """Look for all the labels in the dataset.

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [i[3] for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))
        return

    def get_weights(self):
        """Count number of instances for each class, and computes weights."""

        # Get classes
        self.process_classes()
        all_labels = [self.class_to_idx_[i[3]] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        # logging.debug("Weights: {}".format(self.weights_))
        return self.weights_


    def replicate_balance_instances(self):
        """resample the instances list to balance each class."""
        all_labels = [i[3] for i in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [i for i in self.instances_ if i[1] == l]
            random.shuffle(instances_l)
            instances_l = instances_l * (val_sample // len(instances_l) + 1)
            all_instances_.extend(sorted(instances_l[:val_sample]))

        self.instances_ = all_instances_
        return


    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve patches from patient as specified by idx"""
        # print(f'idx {idx}')
        # print(f'self instances {len(self.instances_)}')
        # patient, target = self.instances_[idx]
        slide_path , patch_idx , patch_tensor, target = self.instances_[idx]
        # print(f'slide_path {slide_path}')
        patch_tensor = patch_tensor.permute(2,0,1)
        
        if target=='lgg':
            target = 0
        else:
            target = 1
        
        # print(f'PATCH {slide_path} , {patch_idx} , {patch_tensor.shape}, {target}')

    

        if self.transform_ is not None:
            im = self.transform_(patch_tensor)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "slide_path": slide_path, "patch_idx":patch_idx}

    def __len__(self):
        return len(self.instances_)


    def reset_index(self):
        """Reset the index of the dataset instances."""
        self.instances_ = list(pd.DataFrame(self.instances_).reset_index(drop=True).to_records(index=False))
