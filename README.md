# SCOD: Sketching Curvature for Out-of-Distribution Detection

This repository contains a PyTorch implementation of the technique described in our paper:

> Sharma, Apoorva, Navid Azizan, and Marco Pavone. "Sketching Curvature for Efficient Out-of-Distribution Detection for Deep Neural Networks." [arXiv preprint arXiv:2102.12567](https://arxiv.org/abs/2102.12567) (2021).

## Overview

This repository provides a framework for wrapping a pre-trained neural network with uncertainty estimates. It is designed to work with any pytorch model. We implement several such wrappers in a general framework. Given a pretrained DNN `model : torch.nn.Module`, the distribution that the network parameterizes `dist_fam : nn_ood.distributions.DistFam`, and a PyTorch dataset containing the training data `dataset : torch.utils.data.Dataset`, we can construct a uncertainty-equipped version of the network as follows:

```
unc_model = UncWrapper(model, dist_fam, **kwargs)
unc_model.process_dataset(dataset)
```

where wrapper specific hyperparameters are passed in as keyword arguments.

Then, we can use `unc_model` as we would use `model,` except the wrapped model now will produce an uncertainty score along with the normal model output:

```
output, unc = unc_model(input)
```

We implement several such uncertainty wrappers, available in `nn_ood.posteriors`:

- `SCOD`: Sketching Curvature for OoD Detection
- `LocalEnsemble`: Implements the algorithm described in [(Madras et al., 2019)](https://arxiv.org/abs/1910.09573)
- `KFAC`: Implements the algorithm described in [(Ritter et al., 2019)](https://arxiv.org/abs/1612.01474)
- `Naive`: Uses the model's output directly to compute an uncertainty score (e.g., entropy of output distribution)

We also compare to DeepEnsembles, which operate on a collection of models of identical architecture. Implementing Deep Ensembles in this framework is a similar process -- after having trained K models, we can intialized the wrapper with a list containing these models
```
models = [model1, model2, ..., modelK]
unc_model = Ensemble(models, dist_fam, **kwargs)
```

## How to use

### Downloading / installing dependencies

Clone this repo (including the submodules):

```
git clone --recurse-submodules git@github.com:StanfordASL/SCOD.git
```

Install the framework (this will autoinstall the required submodules)

```
pip install -e .
```

Download datasets (dataloaders expect data to be in ~/datasets). This script downloads the data for Wine and TinyImagenet. The other datasets used in these experiments are all downloaded automatically through pytorch.
```
./download_datasets.sh -t ~/datasets/
```
Make sure to update `nn_ood/__init__.py` to match the location of your dataset directory. 

### Running experiments
Each domain / experiment has a folder in `experiments` which contains a `config.py` file. This file defines all experiment specific details -- hyperparameters, which dataset to use, model architecture, functions to plot data, etc. The config file also defines which combinations of uncwrappers and hyperparameters to test.

The notebook `experiments/run_experiments.ipynb` has scripts which run experiments as defined by this config file. At the start of the script, replace `EXP_FOLDER` to point to the desired experiment folder so that the correct config.py file is used.

The notebook `experiments/visualize.ipynb` has code that generates plots from the results that are saved by the `run_experiments.ipynb` notebook. The notebook generates experiment-specific plots as defined in the appropriate `EXP_FOLDER/config.py`.

