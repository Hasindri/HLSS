# Hierarchical Discriminative (HiDisc) Learning Improves Visual Representations of Biomedical Microscopy

Companion code for HiDisc, paper in CVPR 2023.

[**HiDisc Website**](https://hidisc.mlins.org) /
[**arXiv**](https://arxiv.org/abs/2303.01605) /
[**MLiNS Lab**](https://mlins.org) /
[**OpenSRH**](https://opensrh.mlins.org)

## HiDisc Overview

![Overview](/figures/positive_pairs.png)

Motivated by the patient-slide-patch data hierarchy of clinical biomedical
microscopy, HiDisc defines a patient, slide, and patch discriminative learning
objective to improve visual representations. Because WSI and microscopy data
are inherently hierarchical, defining a unified hierarchical loss function does
not require additional annotations or supervision. Positive patch pairs are
defined based on a common ancestry in the data hierarchy. A major advantage of
HiDisc is the ability to define positive pairs _without_ the need to sample
from or learn a set of strong image augmentations, such as random erasing,
shears, color inversion, etc. Because each field-of-view in a WSI is a
different view of a patient's underlying cancer diagnosis, HiDisc implicitly
learns image features that predict that diagnosis.

## Visualization of learned SRH representations

![tSNE Plots](/figures/tsne.png)

**Top.** Randomly sampled patch representations are visualized after SimCLR
versus HiDisc pretraining using tSNE. Representations are colored based on
brain tumor diagnosis. HiDisc qualitatively achieves higher quality feature
learning and class separation compared to SimCLR. Expectedly, HiDisc shows
within- diagnosis clustering that corresponds to patient discrimination.
**Bottom.** Magnified cropped regions of the above visualizations show
subclusters that correspond to individual patients. Patch representations in
magnified crops are colored according to patient membership. We see patient
discrimination within the different tumor diagnoses. Importantly, we do not see
patient discrimination within normal brain tissue because there are
minimal-to-no differentiating microscopic features between patients. This
demonstrates that in the absence of discriminative features at the slide- or
patient-level, HiDisc can achieve good feature learning using patch
discrimination without overfitting the other discrimination tasks.

## Installation

1. Clone HiDisc github repo
   ```console
   git clone git@github.com:MLNeurosurg/hidisc.git
   ```
2. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
3. Create conda environment
    ```console
    conda create -n hidisc python=3.10
    ```
4. Activate conda environment
    ```console
    conda activate hidisc
    ```
5. Install package and dependencies
    ```console
    <cd /path/to/hidisc/repo/dir>
    pip install -e .
    ```

## Directory organization
```
hidisc/
├── hidisc/             # library for HiDisc training with OpenSRH
│   ├── datasets/       # PyTorch datasets to work with OpenSRH
│   ├── losses/         # HiDisc loss functions with contrastive learning
│   ├── models/         # PyTorch models for training and evaluation
│   └── train/          # Training and evaluation scrpits
│       └── config/     # Configuration files used for training
├── figures/            # Figures in the README file
├── README.md
├── setup.py            # Setup file including list of dependencies
├── LICENSE             # MIT license for the repo
└── THIRD_PARTY         # License information for third party code
```

## Training / evaluation instructions

The code base is written using PyTorch Lightning, with custom network and
datasets for OpenSRH.

To train HiDisc on the OpenSRH dataset:

1. Download OpenSRH - request data [here](https://opensrh.mlins.org).
2. Update the sample config file in `train/config/train_hidisc.yaml` with
    desired configurations.
3. Change directory to `train` and activate the conda virtual environment.
4. Use `train/train_hidisc.py` to start training:
    ```console
    python train_hidisc.py -c=config/train_hidisc_patient.yaml
    ```

To evaluate with your trained model:
1. Update the sample config file in `train/config/eval.yaml` with
    the checkpoint path and other desired configurations.
2. Change directory to `train` and activate the conda virtual environment.
3. Use `train/train_hidisc.py` for knn evaluation:
    ```console
    python eval_knn.py -c=config/eval.yaml
    ```
