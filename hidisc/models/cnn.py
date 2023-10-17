"""Model wrappers.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Dict, List
from itertools import chain
import open_clip
import torch
from torch import nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from PIL import Image
import os
import json
import logging
from collections import Counter
from typing import Optional, List, Union, TypedDict, Tuple
import random
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

class MLP(nn.Module):
    """MLP for classification head.

    Forward pass returns a tensor.
    """

    def __init__(self, n_in: int, hidden_layers: List[int],
                 n_out: int) -> None:
        super().__init__()
        layers_in = [n_in] + hidden_layers
        layers_out = hidden_layers + [n_out]

        layers_list = list(
            chain.from_iterable((nn.Linear(a, b), nn.ReLU())
                                for a, b in zip(layers_in, layers_out)))[:-1]
        self.layers = nn.Sequential(*layers_list)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class CLIPTextClassifier(nn.Module):
    """Text classifier head whose basis vectors are derived from CLIP text embeddings of classes given in config file.

    Forward pass returns a tensor.
    """

    def __init__(self, arch:str,labels: Dict[str, str], templates: List[str],device = 'cuda') -> None:
        super().__init__()

        model, pct, pv = open_clip.create_model_and_transforms(arch)
        tokenizer = open_clip.get_tokenizer(arch)
        
        zeroshot_weights = []

        for classname in labels.values():
            # print(f'classname {classname}')
            texts = [template.format(c=classname) for template in templates]
            texts = tokenizer(texts) # tokenize
            class_embeddings = model.encode_text(texts)
            # print(f'class embeddings {class_embeddings.shape}')
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            # print(f'class embedding {class_embedding.shape}')
            zeroshot_weights.append(class_embedding)
                    
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

        self.device = device
       

        self.zeroshot_weights = zeroshot_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.zeroshot_weights = self.zeroshot_weights.to(self.device)
        return (100. * x @ self.zeroshot_weights)


class ContrastiveLearningNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj: callable):
        super(ContrastiveLearningNetwork, self).__init__()
        self.bb = backbone()
        self.proj = proj()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x {x.shape}')
        bb_out = self.bb(x)
        # print(f'bb out {bb_out.shape}')
        #bb_out_norm = torch.nn.functional.normalize(bb_out, p=2.0, dim=1)
        proj_out = self.proj(bb_out)
        proj_out_norm = torch.nn.functional.normalize(proj_out, p=2.0, dim=1)
        # print(f'proj_out_norm {proj_out_norm.unsqueeze(1).shape}')

        # print(f"Is proj_out_norm on CUDA/GPU? {proj_out_norm.is_cuda}")
        # print(f"Data type (dtype) of proj_out_norm elements: {proj_out_norm.dtype}")


        return proj_out_norm.unsqueeze(1)
    

class HLSSContrastiveLearningNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj: callable):
        super(HLSSContrastiveLearningNetwork, self).__init__()
        self.bb = backbone()
        self.proj = proj()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x {x.shape}')
        bb_out = self.bb(x)
        # print(f'bb out {bb_out.shape}')
        #bb_out_norm = torch.nn.functional.normalize(bb_out, p=2.0, dim=1)
        proj_out = self.proj(bb_out)
        proj_out_norm = torch.nn.functional.normalize(proj_out, p=2.0, dim=1)
        # print(f'proj_out_norm {proj_out_norm.unsqueeze(1).shape}')

        # print(f"Is proj_out_norm on CUDA/GPU? {proj_out_norm.is_cuda}")
        # print(f"Data type (dtype) of proj_out_norm elements: {proj_out_norm.dtype}")


        return proj_out_norm.unsqueeze(1)
    

class CLIPVisual(nn.Module):
    """Visual encoder from CLIP, given in config file.

    Forward pass returns a tensor.
    """

    def __init__(self, arch:str,device='cuda') -> None:
        super().__init__()

        model, pct, pv = open_clip.create_model_and_transforms(arch)
        self.device = device
        self.model = model.to(device)
        self.traintransform = pct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'visual input {x.shape}')

        x = [transforms.ToPILImage()(image) for image in x]
        x = torch.stack([self.traintransform(image) for image in x]) 
        # print(f'transofrmed x {x.shape}')
        x = x.to(self.device)
        x = self.model.encode_image(x, normalize=True)
        # print(f'image_features {x.shape}')

        return x
    

