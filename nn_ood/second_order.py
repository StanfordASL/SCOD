import torch
import numpy as np
from torch import nn
import scipy.linalg
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh

def zero_grads(model):
    """
    zeros all stored gradients in model
    setting to None advised here: 
        - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    """
    for p in model.parameters():
            if p.grad is not None:
                p.grad = None
