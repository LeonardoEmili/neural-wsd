<h1 align="center">
  Neural WSD
</h1>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
  <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
  <a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
  <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>
</p>

This repo contains our implementation of a simple Word-Sense-Disambiguation system that users can access according to the Software-as-a-service model. We leverage the [PyTorch Lightning Template](https://github.com/edobobo/p-lightning-template) to avoid writing boilerplate code and promote code quality, and [GitHub Actions](https://docs.github.com/en/actions/automating-builds-and-tests/about-continuous-integration) to encourage best practices when deploying to production builds. All the experiments are logged using [Weights & Biases](https://wandb.ai/site). To have a quick overview of the status of this project visit the related [project board](https://github.com/LeonardoEmili/neural-wsd/projects/1).

## Project documentation
The documentation related to this project is available in the [Wiki](https://github.com/LeonardoEmili/neural-wsd/wiki).

## Running the project
For the training procedure, simply run `python -m src.train model.use_lexeme_mask=True`, which will train the model with the default set of hyper-parameters. For the unit tests, run `python -m unittest -v <TEST_SCRIPT>` pointing to the test you are looking for.

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
| tests
  | unit                    # contains unit tests
| README.md
| pyproject.toml            # black configuration file
| requirements.txt
| setup.sh                  # environment setup script 
```

## Authors (contributed equally)
* **Andrea Bacciu**  - [Personal website](https://andreabac3.github.io)
* **Leonardo Emili**  - [Personal website](https://leonardoemili.github.io)
