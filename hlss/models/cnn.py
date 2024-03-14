"""Model wrappers.

Copyright (c) 2024 Mohamed Bin Zayed University of Artificial Intelligence. All rights reserved.
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

    def __init__(self, n_in: int, hidden_layers: List[int],n_out: int, 
                 arch:str,labels: Dict[str, str], templates: List[str],device = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()

        #init MLP
        layers_in = [n_in] + hidden_layers
        layers_out = hidden_layers + [n_out]

        layers_list = list(
            chain.from_iterable((nn.Linear(a, b), nn.ReLU())
                                for a, b in zip(layers_in, layers_out)))[:-1]
        self.layers = nn.Sequential(*layers_list)

        #init mlp weights
        model, pct, pv = open_clip.create_model_and_transforms(arch,device='cpu', pretrained="/data1/dri/hlss/hlss/models/rn50-quickgelu-cc12m-f000538c.pt")
        tokenizer = open_clip.get_tokenizer(arch)
        
        zeroshot_weights = []

        # for classname in labels:
        for classname in templates.keys():

            texts = templates[classname]

            texts = tokenizer(texts) # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        del model
                    
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

        self.device = device
        self.zeroshot_weights = zeroshot_weights

        if hasattr(self, 'zeroshot_weights'):
            for i,linear_layer in enumerate(self.layers):
        
                with torch.no_grad():
                    linear_layer.weight.copy_(self.zeroshot_weights.t())
                    linear_layer.bias.zero_()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ContrastiveLearningNetwork(torch.nn.Module):
    """A network consists of a backbone and projection head.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj: callable):
        super(ContrastiveLearningNetwork, self).__init__()
        self.bb = backbone()
        self.proj = proj()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        bb_out = self.bb(x)
        proj_out = self.proj(bb_out)
        proj_out_norm = torch.nn.functional.normalize(proj_out, p=2.0, dim=1)



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
        bb_out = self.bb(x)
        proj_out = self.proj(bb_out)
        proj_out_norm = torch.nn.functional.normalize(proj_out, p=2.0, dim=1)

        return proj_out_norm.unsqueeze(1)


class HLSSHidiscNetwork(torch.nn.Module):
    """A network consists of a backbone and 2 projection heads.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj1: callable, proj2: callable):
        super(HLSSHidiscNetwork, self).__init__()
        self.bb = backbone()
        self.proj1 = proj1()
        self.proj2 = proj2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb_out = self.bb(x)

        #CLIPTextClassifier
        proj_out1 = self.proj1(bb_out)
        #MLP
        proj_out2 = self.proj2(bb_out)

        proj_out_norm1 = torch.nn.functional.normalize(proj_out1, p=2.0, dim=1)
        proj_out_norm2 = torch.nn.functional.normalize(proj_out2, p=2.0, dim=1)

        return proj_out_norm1.unsqueeze(1), proj_out_norm2.unsqueeze(1)
    
class ResnetHLSSNetwork(torch.nn.Module):
    """A network consists of a backbone and 2 projection heads.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj1: callable, proj2: callable):
        super(ResnetHLSSNetwork, self).__init__()
        self.bb = backbone()
        self.proj1 = proj1()
        self.proj2 = proj2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb_out = self.bb(x)

        #MLP
        proj_out1 = self.proj1(bb_out)
        #CLIPTextClassifier
        proj_out2 = self.proj2(proj_out1)

        proj_out_norm2 = torch.nn.functional.normalize(proj_out2, p=2.0, dim=1)

        return proj_out_norm2.unsqueeze(1)


class three_MLP(torch.nn.Module):
    """A network consists of a backbone and 3 MLP projection heads.

    Forward pass returns the normalized embeddings after projection layers.
    """

    def __init__(self, backbone: callable, proj1: callable, proj2: callable, proj3: callable):
        super(three_MLP, self).__init__()
        self.bb = backbone()
        self.proj1 = proj1()
        self.proj2 = proj2()
        self.proj3 = proj3()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb_out = self.bb(x)

        #MLP1
        proj_out1 = self.proj1(bb_out)
        #MLP2
        proj_out2 = self.proj2(bb_out)
        #MLP3
        proj_out3 = self.proj3(bb_out)

        proj_out_norm1 = torch.nn.functional.normalize(proj_out1, p=2.0, dim=1)
        proj_out_norm2 = torch.nn.functional.normalize(proj_out2, p=2.0, dim=1)
        proj_out_norm3 = torch.nn.functional.normalize(proj_out3, p=2.0, dim=1)

        return proj_out_norm1.unsqueeze(1), proj_out_norm2.unsqueeze(1), proj_out_norm3.unsqueeze(1)
    

class CLIPVisual(nn.Module):
    """Visual encoder from CLIP, given in config file.

    Forward pass returns a tensor.
    """

    def __init__(self, arch:str,device="cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()

        model, pct, pv = open_clip.create_model_and_transforms(arch,device=device, pretrained="/data1/dri/hlss/hlss/models/rn50-quickgelu-cc12m-f000538c.pt")
        
        self.device = device
        self.model = model.to(device)
        self.traintransform = pct


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.to(self.device)
        x = self.model.encode_image(x, normalize=True)

        return x
    

class HLSSGranularNetwork(torch.nn.Module):
    """A network consists of a backbone and 32 projection heads.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj1: callable, proj2: callable, proj3: callable):
        super(HLSSGranularNetwork, self).__init__()
        self.bb = backbone()
        self.proj1 = proj1()
        self.proj2 = proj2()
        self.proj3 = proj3()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb_out = self.bb(x)

        #CLIPTextClassifier patch
        proj_out1 = self.proj1(bb_out)
        #CLIPTextClassifier slide
        proj_out2 = self.proj2(bb_out)
        #CLIPTextClassifier patient
        proj_out3 = self.proj3(bb_out)

        proj_out_norm1 = torch.nn.functional.normalize(proj_out1, p=2.0, dim=1)
        proj_out_norm2 = torch.nn.functional.normalize(proj_out2, p=2.0, dim=1)
        proj_out_norm3 = torch.nn.functional.normalize(proj_out3, p=2.0, dim=1)

        return proj_out_norm1.unsqueeze(1), proj_out_norm2.unsqueeze(1), proj_out_norm3.unsqueeze(1)
    

class HLSSKL(torch.nn.Module):
    """A network consists of a backbone and 3 projection heads.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone: callable, proj1: callable, proj2: callable, proj3: callable):
        super(HLSSKL, self).__init__()
        self.bb = backbone()
        self.proj1 = proj1()
        self.proj2 = proj2()
        self.proj3 = proj3()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb_out = self.bb(x)

        #CLIPTextClassifier patch
        proj_out1 = self.proj1(bb_out)
        #CLIPTextClassifier slide
        proj_out2 = self.proj2(bb_out)
        #CLIPTextClassifier patient
        proj_out3 = self.proj3(bb_out)

        bb_out_norm = torch.nn.functional.normalize(bb_out, p=2.0, dim=1)
        proj_out_norm1 = torch.nn.functional.normalize(proj_out1, p=2.0, dim=1)
        proj_out_norm2 = torch.nn.functional.normalize(proj_out2, p=2.0, dim=1)
        proj_out_norm3 = torch.nn.functional.normalize(proj_out3, p=2.0, dim=1)

        return bb_out_norm.unsqueeze(1),proj_out_norm1.unsqueeze(1), proj_out_norm2.unsqueeze(1), proj_out_norm3.unsqueeze(1)
    

class HLSSGranularKL(torch.nn.Module):
    """A network consists of a 2 backbones and 3 projection heads.

    Forward pass returns the normalized embeddings after a projection layer.
    """

    def __init__(self, backbone1: callable, backbone2: callable,  proj1: callable, proj2: callable, proj3: callable):
        super(HLSSGranularKL, self).__init__()
        self.visualbb = backbone1()
        self.textbb = backbone2()
        self.proj1 = proj1()
        self.proj2 = proj2()
        self.proj3 = proj3()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vis_out = self.visualbb(x)
        txt_out = self.textbb(x)

        #CLIPTextClassifier patch
        proj_out1 = self.proj1(txt_out)
        #CLIPTextClassifier slide
        proj_out2 = self.proj2(txt_out)
        #CLIPTextClassifier patient
        proj_out3 = self.proj3(txt_out)

        vis_out_norm = torch.nn.functional.normalize(vis_out, p=2.0, dim=1)
        proj_out_norm1 = torch.nn.functional.normalize(proj_out1, p=2.0, dim=1)
        proj_out_norm2 = torch.nn.functional.normalize(proj_out2, p=2.0, dim=1)
        proj_out_norm3 = torch.nn.functional.normalize(proj_out3, p=2.0, dim=1)
 

        return vis_out_norm.unsqueeze(1), proj_out_norm1.unsqueeze(1), proj_out_norm2.unsqueeze(1), proj_out_norm3.unsqueeze(1)