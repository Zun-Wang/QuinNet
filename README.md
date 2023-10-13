<div align="center">

# QuinNet


</div>

<br>

## 📌  Introduction

QuinNet is an equivariant GNN architecture which incorporates 3-body, 4-body, and 5-body interactions, for constructing force fields for molecular dynamics simulations. Details could be found in [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/71146).

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
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install pytorch-lightning==1.8.3
```

Train QuinNet

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --conf example/hparams_aspirin.yaml
```


## Acknowledgements
This project is based on the repo [torchmd-net](https://github.com/torchmd/torchmd-net.git).