import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from nn_ood.data.imagenet import ImageNetData
from nn_ood.posteriors import ExtrapUncertainty, SWAGS, Ensemble, Naive, KFAC
from nn_ood.distributions import CategoricalLogit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 5

LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9

LR_DROP_FACTOR = 0.5
EPOCHS_PER_DROP = 2

BATCH_SIZE = 5

N_EPOCHS = 20

## SET UP DATASETS
dataset_class = ImageNetData
test_dataset_args = ['val', 'ood']

## DEFINE VISUALIZATION FUNCTIONS
def plt_image(ax, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return ax.imshow(inp)
    
def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    input, target = dataset[idx]
    plt_image(ax, input)
    xlabel = 'Target: %d' % target
    if unc_model is not None:
        input = input.to(device)
        pred, unc = unc_model(input.unsqueeze(0))
        pred = pred[0].detach().cpu().numpy()
        unc = unc.item()
        xlabel += '\nPred: %d\nUnc: %0.3f' % (np.argmax(pred), unc)
    elif model is not None:
        input = input.to(device)
        pred = model(input.unsqueeze(0))[0].detach().cpu().numpy()
        xlabel += '\nPred: %d' % np.argmax(pred)
        print(pred)
        
    ax.set_xlabel(xlabel)


def viz_datasets(idx=0, unc_model=None, model=None):
    num_plots = len(test_dataset_args)
    fig, axes = plt.subplots(1,num_plots, figsize=[5*num_plots, 5], dpi=100)
    for i, split in enumerate( test_dataset_args ):
        dataset = dataset_class(split)
        viz_dataset_sample(axes[i], dataset, idx=idx, unc_model=unc_model, model=model)

## USE CUDA IF POSSIBLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# henofthewoods, mushroom, stinkhorn, coralfungus
# classesToKeep = [996, #947, 
#                  994, 991]

# stbernard, bagel, jeep
classesToKeep = [247, 931, 609]

class classSelector(nn.Module):
    def __init__(self, idx_list):
        super().__init__()
        self.idx_list = idx_list
        
    def forward(self,x):
        return x[..., self.idx_list]

## MODEL SET UP
og_req_grads = []
def make_model(freeze=False):
    global og_req_grads
    resnet_base = models.resnet18(pretrained=True)
    selector = classSelector(classesToKeep)
    model = nn.Sequential(resnet_base, selector)
    
    og_req_grads = [p.requires_grad for p in model.parameters()]
    
    return model

def freeze_model(model):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # make last layer trainable
    for p in model.fc.parameters():
        p.requires_grad = True
        
def unfreeze_model(model):
    global og_req_grads
    # unfreeze everything
    for p,v in zip( model.parameters(), og_req_grads):
        p.requires_grad = v

def disable_batchnorm(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for n,p in m.named_parameters():
            p.requires_grad_(False)
            
## OPTIMIZATION 
dist_fam = CategoricalLogit().to(device) #criterion = nn.MSELoss()
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

# recipe for preparing uncertainty models
prep_unc_models = {
#     'extrap': {
#         'class': ExtrapUncertainty,
#         'kwargs': {
#             'batch_size': 32,
#             'max_samples': 4,
#             'num_eigs': 20,
#             'device': 'gpu'
#         }
#     },
#     'swags_SRFTone_s40_n20': {
#         'class': SWAGS,
#         'kwargs': {
#             'num_samples': 40,
#             'num_eigs': 20,
#             'device':'gpu',
#             'sketch_type': 'srft_onesided'
#         },
#     },
    'swags_SRFT_s40_n10': {
        'class': SWAGS,
        'kwargs': {
            'num_samples': 40,
            'num_eigs': 10,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
    'kfac': {
        'class': KFAC,
        'kwargs': {
            'batch_size': 16,
            'device':'gpu',
            'input_shape': [3,224,224]
        },
    },
}

# recipe for testing uncertainty models
test_unc_models = {
    'extrap': {
        'class': ExtrapUncertainty,
        'kwargs': {
            'num_eigs': 20,
            'device': 'gpu',
            'n_y_samp': 3,
        },
        'load_name': 'extrap',
        'forward_kwargs': {
            'n_samples': 1
        }
    },
    'GBAD': {
        'class': SWAGS,
        'kwargs': {
            'num_eigs': 10,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'normalize':False,
            'device':'gpu',
        },
        'load_name': 'swags_SRFT_s40_n10',
        'forward_kwargs': {}
    },
#     'swags_SRFTone_s40_n20': {
#         'class': SWAGS,
#         'kwargs': {
#             'num_eigs': 20,
#             'batch_size': 4,
#             'learning_rate': 1e-5,
#             'normalize':False,
#             'device':'gpu',
#         },
#         'load_name': 'swags_SRFTone_s40_n20',
#         'forward_kwargs': {}
#     },
    'kfac': {
        'class': KFAC,
        'kwargs': {
            'device':'gpu',
            'input_shape':[3,224,224]
        },
        'load_name': 'kfac',
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
            'device': 'gpu',
        },
        'load_name': None,
        'multi_model': True,
        'forward_kwargs': {}
    }
}

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1.

in_dist_splits = test_dataset_args[0:1]
out_dist_splits = test_dataset_args[1:]

# Visualization
keys_to_compare =  [
                    'GBAD',
                    'ensemble', 
                    'extrap',
                    'kfac',
                    'naive',
                   ]

colors= [
#          'xkcd:olive',
         'xkcd:teal',
#          'xkcd:green',
#          'xkcd:darkgreen',
         'xkcd:darkblue',
#          'xkcd:aquamarine',
         'xkcd:red',
         'xkcd:maroon',
         'xkcd:orange',
         'xkcd:coral',
         'xkcd:blue',
         'xkcd:purple',
         'xkcd:violet']