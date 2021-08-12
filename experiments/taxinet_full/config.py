import torch
import torch.nn as nn
from torchvision import models
from nn_ood.data.taxinet import TaxiNetFull
from nn_ood.posteriors import LocalEnsemble, Ensemble, SCOD, KFAC, Naive
from nn_ood.distributions import GaussianFixedDiagVar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 5

LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9

LR_DROP_FACTOR = 0.5
EPOCHS_PER_DROP = 2

BATCH_SIZE = 5

N_EPOCHS = 10

## SET UP DATASETS
dataset_class = TaxiNetFull
test_dataset_args = ['exp1_train', 'exp2_train', 'exp3_train', 'exp4_train', 'exp5_train']

## DEFINE VISUALIZATION FUNCTIONS
def plt_image(ax, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return ax.imshow(inp)
    
def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    input, target = dataset[idx]
    plt_image(ax, input)
    xlabel = 'Target: %0.2f,%0.2f' % (target[0].item(), target[1].item())
    if unc_model is not None:
        input = input.to(device)
        pred, unc = unc_model(input.unsqueeze(0))
        pred = pred[0].cpu().detach().numpy()
        unc = unc.item()
        xlabel += '\nPred: %02f,%02f\nUnc: %0.3f' % (pred[0],pred[1], unc)
    elif model is not None:
        input = input.to(device)
        pred = model(input.unsqueeze(0))[0].cpu().detach().numpy()
        xlabel += '\nPred: %0.3f' % (pred[0],pred[1])
     
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
og_req_grads = []
def make_model():
    global og_req_grads
    model = models.resnet18(pretrained=True)
    y_dim = 2
    model.fc = nn.Linear(model.fc.in_features, y_dim)
    og_req_grads = [p.requires_grad for p in model.parameters()]
    
    return model

def freeze_model(model, freeze_frac=True):
    # freeze everything
    n_params = len(list(model.parameters()))
    for i, p in enumerate(model.parameters()):
        if i < 6*n_params/7:
            p.requires_grad = False

    # make last layer trainable
    for p in model.fc.parameters():
        p.requires_grad = True
        
def unfreeze_model(model):
    global og_req_grads
    # unfreeze everything
    for p,v in zip( model.parameters(), og_req_grads):
        p.requires_grad = v
    
## OPTIMIZATION 
dist_fam = GaussianFixedDiagVar(sigma_diag=np.array([1., 1.])).to(device)
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
            'batch_size': 32,
            'max_samples': 4,
            'num_eigs': 14,
            'device':'gpu'
        },
    },
    'kfac': {
        'class': KFAC,
        'kwargs': {
            'device':'gpu',
            'input_shape': [3, 200, 360],
            'num_loss_samples': 5,
        },
    },
    'scod_SRFT_s124_n20_freeze': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 124,
            'num_eigs': 20,
            'device':'gpu',
            'sketch_type': 'srft'
        },
        'freeze':True
    },
    'scod_SRFT_s46_n7': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 46,
            'num_eigs': 7,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
}

test_unc_models = {
    'local_ensemble': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 14,
            'device':'gpu',
            'n_y_samp': 1,
        },
        'load_name': 'local_ensemble',
        'forward_kwargs': {}
    },
    'kfac_n1_s100': {
        'class': KFAC,
        'kwargs': {
            'device':'gpu',
            'input_shape': [3, 200, 360],
        },
        'load_name': 'kfac',
        'forward_kwargs': {
            'norm': 1000.,
            'scale': 10000.
        }
    },
    'SCOD (T=46,k=7)': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 46,
            'num_eigs': 7,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'proj_type':'posterior_pred',
            'device':'gpu'
        },
        'load_name': 'scod_SRFT_s46_n7',
        'forward_kwargs': {
            'n_eigs':7,
        }
    },
    'SCOD_freeze (k=20)': {
        'class': SCOD,
        'kwargs': {
            'num_samples':124,
            'num_eigs': 20,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'proj_type':'posterior_pred',
            'device':'gpu'
        },
        'freeze':True,
        'load_name': 'scod_SRFT_s124_n20_freeze',
        'forward_kwargs': {
            'n_eigs': 20,
        }
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
splits_to_use = test_dataset_args[1:2]
err_thresh = 1.

in_dist_splits = test_dataset_args[0:1]
out_dist_splits = test_dataset_args[2:]


# Visualization
from nn_ood.utils.viz import summarize_ood_results, summarize_ood_results_by_error, plot_perf_vs_runtime

keys_to_compare = [
    'SCOD (T=46,k=7)',
    'SCOD_freeze (k=20)',
    'ensemble', 
    'local_ensemble',
    'kfac_n1_s100',
    'naive',
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
        'title': "TaxiNet",
    },
    'error_based.pdf': {
        'summary_fn': summarize_ood_results_by_error,
        'summary_fn_args': [
            splits_to_use,
            err_thresh,
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
        'title': "TaxiNet (error based)",
    }
}


# ANIMATION UTILS
class Animator:
    def __init__(self, unc_model, split, label, start_idx, end_idx, past_uncs=[], **kwargs):
        fig, axes = plt.subplots(1,2,figsize=[11, 4],gridspec_kw={
            'width_ratios': [7, 4],
        })
        
        self.fig = fig
        self.axes = axes
        
        self.unc_model = unc_model
        
        self.dataset = dataset_class(split, N=end_idx+1, shuffle=False)
        
        self.start = start_idx
        self.end = end_idx
        
        self.ts = np.arange(self.end-self.start)
        self.errs = [np.nan]*len(self.ts)
        self.uncs = [np.nan]*len(self.ts)
        self.high_err_uncs = [np.nan]*len(self.ts)
        
        self.im = plt_image(self.axes[0], self.dataset[self.start][0])
        
        handles = []
        labels = []
        for data in past_uncs:
            h, = self.axes[1].plot(self.ts, data['uncs'], linestyle=':', color=data['color'])
            labels.append(data['label'])
            handles.append(h)
        
        self.uncplot, = self.axes[1].plot(self.ts, self.uncs, **kwargs)
        handles.append(self.uncplot)
        labels.append(label)
        self.errplot, = self.axes[1].plot(self.ts, self.uncs, marker='x', linestyle='', color='black')
        handles.append(self.errplot)
        labels.append(" ↪ error > 0.05m")
        self.axes[1].legend(handles=handles, labels=labels) 
        
        axes[0].set_title(label)
        axes[0].axis('off')
        axes[1].set_xlim([0, self.end-self.start])
        axes[1].set_title("SCOD Uncertainty Measure")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylim(0,50)
        
        fig.tight_layout()
        
    
    def _update_fig(self, i):        
        inp, target = self.dataset[i]
        output, unc = self.unc_model(inp[None,...].cuda())
        
        inp = inp.numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        self.im.set_array(inp)
        
        err = (output[0,0].cpu() - target[0])**2
        self.errs[i-self.start] = err.detach().numpy()
        
        unc = unc[0].cpu().detach().numpy()
        self.uncs[i-self.start] = unc
        torch.cuda.empty_cache()
        
        self.uncplot.set_xdata(self.ts)
        self.uncplot.set_ydata(self.uncs)
        
        if err > 0.05:
            self.high_err_uncs[i - self.start] = unc
        
        self.errplot.set_xdata(self.ts)
        self.errplot.set_ydata(self.high_err_uncs)
        
        return self.im, self.uncplot, self.errplot
    
    def create(self):
        ani = animation.FuncAnimation(self.fig, self._update_fig, frames=range(self.start, self.end), interval=50)
        return ani, self.uncs