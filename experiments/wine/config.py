import torch
import torch.nn as nn
from nn_ood.data.wine import WineData
from nn_ood.posteriors import LocalEnsemble, Ensemble, SCOD, KFAC, Naive
from nn_ood.distributions import GaussianFixedDiagVar
import numpy as np
import matplotlib.pyplot as plt

    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 5

LEARNING_RATE = 0.02

BATCH_SIZE = 5

N_EPOCHS = 20

## SET UP DATASETS
dataset_class = WineData
test_dataset_args = ['val', 'whites', 'random']

def viz_datasets(idx=0, unc_model=None, model=None):
    for split in ['train'] + test_dataset_args:
        dataset = dataset_class(split)
        print(split, ": M =", len(dataset))

## USE CUDA IF POSSIBLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## MODEL SET UP
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=5./3)
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)

def make_model():
    model = nn.Sequential(
        nn.Linear(11, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    model.apply(weight_init)
    
    return model

def freeze_model(model, freeze_frac=True):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # make everything beyond layer k tunable
    k = 1
    n_layers = len(list(model.children()))
    for i, m in enumerate(model.children()):
        if i >= 2*k:
            for p in m.parameters():
                p.requires_grad = True
    
def unfreeze_model(model):
    # unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

        
dist_fam = GaussianFixedDiagVar(sigma_diag=np.array([1.])).to(device)
opt_class = torch.optim.Adam
opt_kwargs = {
    'lr': 0.02,
}
sched_class = torch.optim.lr_scheduler.StepLR
sched_kwargs = {
    'step_size': 20, # epochs per drop in learning rate
    'gamma': 1., # factor of drop
}    
    
prep_unc_models = {
    'local_ensemble': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 20,
            'device':'gpu',
            'full_data':True,
        },
    },
    'kfac': {
        'class': KFAC,
        'kwargs': {
            'input_shape': [11],
            'device':'gpu',
        },
    },
    'scod_SRFT_s604_n100': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 604,
            'num_eigs': 100,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
    'scod_SRFT_s604_n100_freeze': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 604,
            'num_eigs': 100,
            'device':'gpu',
            'sketch_type': 'srft'
        },
        'freeze': True,
    },
}

test_unc_models = {
    'local_ensemble_n20': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 20,
            'device':'gpu',
            'n_y_samp': 1,
        },
        'load_name': 'local_ensemble',
        'forward_kwargs': {}
    },
    'local_ensemble_n10': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 20,
            'device':'gpu'
        },
        'load_name': 'local_ensemble',
        'forward_kwargs': {
            'n_eigs': 10
        }
    },
    'kfac': {
        'class': KFAC,
        'kwargs': {
            'input_shape': [11],
            'device':'gpu'
        },
        'load_name': 'kfac',
        'forward_kwargs': {}
    },
    'SCOD': {
        'class': SCOD,
        'kwargs': {
            'num_eigs': 100,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'proj_type':'posterior_pred',
            'device':'gpu'
        },
        'load_name': 'scod_SRFT_s604_n100',
        'forward_kwargs': {}
    },
    'SCOD_freeze': {
        'class': SCOD,
        'kwargs': {
            'num_eigs': 100,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'proj_type':'posterior_pred',
            'device':'gpu'
        },
        'freeze': True,
        'load_name': 'scod_SRFT_s604_n100_freeze',
        'forward_kwargs': {}
    },
    'naive': {
        'class': Naive,
        'kwargs': {
            'device':'gpu'
        },
        'load_name': None,
        'forward_kwargs': {}
    },
    'ensemble': {
        'class': Ensemble,
        'kwargs': {
            'device':'gpu'
        },
        'load_name': None,
        'multi_model': True,
        'forward_kwargs': {}
    }
}

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1.

in_dist_splits = test_dataset_args[:1]
out_dist_splits = ['whites', 'random']


# Visualization
from nn_ood.utils.viz import summarize_ood_results, plot_perf_vs_runtime

keys_to_compare = [
    'SCOD',
    'SCOD_freeze',
    'ensemble', 
    'local_ensemble_n20',
    'kfac',
    'naive'
]

colors= [
    'xkcd:azure',
    'xkcd:electric blue',
    'xkcd:adobe',
    'xkcd:mango',
    'xkcd:blood orange',
    'xkcd:scarlet',
]

# Plots to generate:
plots_to_generate = {
    'auroc_vs_runtime.pdf': {
        'summary_fn': summarize_ood_results,
        'summary_fn_args': [
            in_dist_splits,
            out_dist_splits
        ],
        'summary_fn_kwargs': {
            'keys_to_compare': keys_to_compare,
        },
        'plot_fn': plot_perf_vs_runtime,
        'plot_fn_args': [],
        'plot_fn_kwargs': {
            'colors': colors,
            'figsize': [4,2.5],
            'dpi': 150,
            'normalize_x': True,
        },
        'legend': {},
        'title': "Wine",
    },
}