import torch
import torch.nn as nn
from nn_ood.data.binary_mnist import BinaryMNIST
from nn_ood.posteriors import LocalEnsemble, Ensemble, SCOD, Naive, KFAC
from nn_ood.distributions import BernoulliLogit
import numpy as np
import matplotlib.pyplot as plt
    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 5

LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9

LR_DROP_FACTOR = 0.5
EPOCHS_PER_DROP = 5

BATCH_SIZE = 16

N_EPOCHS = 5

## SET UP DATASETS
dataset_class = BinaryMNIST
test_dataset_args = ['val', 'ood', 'fashion']

## DEFINE VISUALIZATION FUNCTIONS
def plt_image(ax, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = 0.1307
    std = 0.3081
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp[:,:,0], cmap='Greys')
    
def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    print(len(dataset))
    input, target = dataset[idx]
    plt_image(ax, input)
    xlabel = 'Target: %d' % target
    if unc_model is not None:
        input = input.to(device)
        pred, unc = unc_model(input.unsqueeze(0))
        pred = pred.item() > 0.5
        unc = unc.item()
        xlabel += '\nPred: %d\nUnc: %0.3f' % (pred, unc)
    elif model is not None:
        input = input.to(device)
        pred = model(input.unsqueeze(0)).item() > 0.5
        xlabel += '\nPred: %d' % pred
     
    ax.set_xlabel(xlabel)
    
def viz_datasets(idx=0, unc_model=None, model=None):
    num_plots = len(test_dataset_args)
    fig, axes = plt.subplots(1,num_plots, figsize=[5*num_plots, 5], dpi=100)
    for i, split in enumerate( test_dataset_args ):
        dataset = dataset_class(split)
        viz_dataset_sample(axes[i], dataset, idx=idx, unc_model=unc_model, model=model)

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
        nn.Conv2d(1, 16, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(288, 1)
    )
    model.apply(weight_init)
    
    return model

def freeze_model(model, freeze_frac=True):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # make everything beyond layer k tunable
    k = 6
    n_layers = len(list(model.children()))
    for i, m in enumerate(model.children()):
        print(i,m)
        if i >= k:
            for p in m.parameters():
                p.requires_grad = True
    
def unfreeze_model(model):
    # unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

dist_fam = BernoulliLogit().to(device)
opt_class = torch.optim.SGD
opt_kwargs = {
    'lr': LEARNING_RATE,
    'momentum': SGD_MOMENTUM
}
sched_class = torch.optim.lr_scheduler.StepLR
sched_kwargs = {
    'step_size': EPOCHS_PER_DROP,
    'gamma': LR_DROP_FACTOR
}    
    
prep_unc_models = {
    'local_ensemble': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 50,
            'device':'gpu',
            'full_data': True,
        },
    },
    'scod_SRFT_s304_n50': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 304,
            'num_eigs': 50,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
    'scod_SRFT_s304_n50_freeze': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 304,
            'num_eigs': 50,
            'device':'gpu',
            'sketch_type': 'srft'
        },
        'freeze': True,
    },
    'kfac': {
        'class': KFAC,
        'kwargs': {
            'device':'gpu',
            'input_shape': [1, 28, 28],
        },
    },
}

test_unc_models = {
    'local_ensemble_n50': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 50,
            'n_y_samp': 2,
            'device':'gpu'
        },
        'load_name': 'local_ensemble',
        'forward_kwargs': {}
    },
    'SCOD': {
        'class': SCOD,
        'kwargs': {
            'num_eigs': 50,
            'device':'gpu'
        },
        'load_name': 'scod_SRFT_s304_n50',
        'forward_kwargs': {}
    },
    'SCOD_freeze': {
        'class': SCOD,
        'kwargs': {
            'num_eigs': 50,
            'device':'gpu'
        },
        'freeze':True,
        'load_name': 'scod_SRFT_s304_n50_freeze',
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
    'kfac_100_10': {
        'class': KFAC,
        'kwargs': {
            'device':'gpu',
            'input_shape': [1, 28, 28]
        },
        'load_name': 'kfac',
        'forward_kwargs': {
            'norm': 100.,
            'scale': 10.,
        }
    },
    'ensemble': {
        'class': Ensemble,
        'kwargs': {
            'device':'gpu'
        },
        'load_name': None,
        'multi_model': True,
        'forward_kwargs': {}
    },
}

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1.

in_dist_splits = ['val']
out_dist_splits = ['ood', 'fashion']


# Visualization
from nn_ood.utils.viz import summarize_ood_results, plot_perf_vs_runtime


keys_to_compare = [
                   'SCOD',
                   'SCOD_freeze',
                   'ensemble', 
                   'local_ensemble_n50',
                   'kfac_100_10',
                   'naive',
                   'maha',
]

colors= [
         'xkcd:azure',
         'xkcd:electric blue',
         'xkcd:adobe',
         'xkcd:mango',
         'xkcd:blood orange',
         'xkcd:scarlet',
         'xkcd:indigo'
]

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
        'title': "Binary MNIST",
    },
}