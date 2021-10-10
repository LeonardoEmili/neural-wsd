<h1 align="center">
  Neural WSD
</h1>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
  <a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
  <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>
</p>

This repo contains all the code used for the assignment of the course Applied Deep Learning at TU Wien. To encourage best practices and promote code quality, we leverage the [PyTorch Lightning Template](https://github.com/edobobo/p-lightning-template) to avoid writing boilerplate code. All the experiments are logged using [Weights & Biases](https://wandb.ai/site). To have a quick overview of the status of this project visit the related [project board](https://github.com/LeonardoEmili/neural-wsd/projects/1).

## Repository Structure
```
neural-wsd
| conf                      # contains Hydra config files
  | data
  | model
  | train
  root.yaml                 # hydra root config file
| data                      # datasets should go here
| experiments               # where the models are stored
| src
  | pl_data_modules.py      # base LightinigDataModule
  | pl_modules.py           # base LightningModule
  | train.py                # main script for training the network
| README.md
| requirements.txt
| setup.sh                  # environment setup script 
```

## Author
> Leonardo Emili - e12109608@student.tuwien.ac.at
