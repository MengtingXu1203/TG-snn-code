# TG-SNN

Code for "On the Role of Temporal Granularity in the Robustness of Spiking Neural Networks (CVPR2026)"

## Prerequisites

The Following Setup is tested and it is working:

- Python>=3.5

- Pytorch>=1.9.0

- Cuda>=10.2

## Data preparation

- CIFAR10: `def build_cifar(use_cifar10=True)` in `data_loaders.py`

- CIFAR100: `def build_cifar(use_cifar10=False)` in `data_loaders.py`
 
 - Tiny-ImageNet: 
 
   (1) Download Tiny-ImageNet dataset
   
   (2)`def build_tiny_imagenet()` in `data_loaders.py`

## Description

- Use a triangle-like surrogate gradient `ZIF` in `layers.py` for step function forward and backward.

## TG-Reg Training

run `bash script/cifar100_vgg11.sh`; 
 