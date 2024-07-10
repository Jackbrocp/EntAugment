import pathlib
import sys
# sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
import argparse
import os
from typing import Generator
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Network import *
from augmentation.cutout import Cutout
from augmentation.entaugment import EntAugment
from augmentation import trivialaugment

def make_magnitude_EntAugment(magnitude, cutout_length):
    trivialaugment.set_augmentation_space(augmentation_space='standard',num_strengths=30)
    magnitude_transform = transforms.Compose([
                transforms.ToPILImage(),
                EntAugment(M=magnitude),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1,cutout_length)
            ])
    return magnitude_transform

def make_transform(length=8, M=6):
    # The augmentation for the warm-up phase (only 10 epochs) can be Cutout or TrivialAugment
    trivialaugment.set_augmentation_space(augmentation_space='standard',num_strengths=30)
    transform = transforms.Compose([
                transforms.ToPILImage(),
                trivialaugment.TrivialAugment(),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
                Cutout(1,length)
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                ])
    return transform, transform_test