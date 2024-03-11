<div align="center">

# QuinNet: Efficiently Incorporating Quintuple Interactions into Geometric Deep Learning Force Fields


</div>

<br>

## 📌  Introduction

This repository contains the source code for NeurIPS 2023 paper "QuinNet: Efficiently Incorporating Quintuple Interactions into Geometric Deep Learning Force Fields". QuinNet is an equivariant graph neural network that efficiently expresses many-body interactions up to five-body interactions with ab initio accuracy.

<br>


## 🚀  Quickstart

Install dependencies

```bash
# clone project
git clone https://github.com/Zun-Wang/QuinNet.git
cd QuinNet

# [OPTIONAL] create conda environment
[Optional] conda create -n QuinNet python=3.9
[Optional] conda activate QuinNet

# Recommed to install part of dependencies in advance
# Take `cuda116` version as an example
pip install rdkit ase
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch_geometric==2.3.0
pip install pytorch-lightning==1.8.3
```

Train QuinNet

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --conf example/hparams_aspirin.yaml
```


## Acknowledgements
This project is based on the repo [torchmd-net](https://github.com/torchmd/torchmd-net.git).
