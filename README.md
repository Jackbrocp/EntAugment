# [ECCV 2024] EntAugment: Entropy-Driven Adaptive Data Augmentation Framework for Image Classification

## Introduction
This is the implementation of EntAugment and EntLoss, as used in the paper. In this paper, we propose a tuning-free and adaptive DA framework, which dynamically assesses and adjusts the augmentation magnitudes for each sample during training.
You can directly start off using our implementations.

## Getting Started
- Codes support Python3

- Clone this directory and `cd`  into it.
 
`git clone https://github.com/Jackbrocp/EntAugment` 

`cd EntAugment`

## Updates
- 2024/7/15: Initial release

## Getting Started
### Requirements
- Python 3
- PyTorch 1.6.0
- Torchvision 0.7.0
- Numpy
<!-- Install a fitting Pytorch version for your setup with GPU support, as our implementation  -->

## Run Data Augmentation
### Prepare the datasets
Download the datasets (e.g., CIFAR datasets) and put the datasets under the folder ```data/```

### Parameters
```--conf```，path to the config file, e.g., ```confs/resnet18.yaml```

### Training Examples
Employ EntAugment as a data augmentation method to train ResNet18 model on CIFAR10 dataset.

```python train_EntAugment.py --conf confs/resnet18.yaml --dataset CIFAR10```

Employ EntAugment and EntLoss to train ResNet18 model on CIFAR100 dataset.

```python train_EntLoss.py --conf confs/resnet18.yaml --dataset CIFAR100```

## Acknowledge 
Part of our implementation is adopted from the [TrivialAugment](https://github.com/automl/trivialaugment) repositories.


## Citation
If you find this repository useful in your research, please cite our paper:

`
To be updated
`
