# MIE-GAT (Multi-perspective Information Enhancement Graph Attention Network)
Code for the paper "MIE-GAT: A Multi-perspective Information Enhancement Graph Attention Network for Automated Lung Nodule Diagnosis in CT Images".
## Requirement
Python 3.9.18

PyTorch == 2.1.1

tensorboardX == 2.6.2.2

numpy == 1.26.3
## Installation
pytorch: http://pytorch.org/

tensorboardX: https://github.com/lanpa/tensorboard-pytorch

Download and unzip this project: MIE-GAT-master.zip.
## Dataset
Original LIDC-IDRI dataset can be found in the official website: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

The LIDP dataset can be found in the paper "Lidp: A lung image dataset with pathological information for
lung cancer screening"
## Todos
* Scripts for the dataset configuration and model hyperparameter configuration are provided in [./config](https://github.com/ly706/MIE-GAT/tree/main/models).
* Scripts for the model architecture are provided in [./models](https://github.com/ly706/MIE-GAT/tree/main/models).
* In the main running scripts, the parameter "mode" is "train" for training mode, "mode" is "eval" for test mode.
* Modify the dataset configuration and the output directory in main running scripts.
* Run main running scripts.
