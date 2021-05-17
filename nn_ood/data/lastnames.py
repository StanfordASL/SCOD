from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import unicodedata
import string
import random
import numpy as np


def findFiles(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('/home/apoorva/datasets/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

######################################################################
# Turning Names into Tensors
# --------------------------
#
# Now that we have all the names organized, we need to turn them into
# Tensors to make any use of them.
#
# To represent a single letter, we use a "one-hot vector" of size
# ``<1 x n_letters>``. A one-hot vector is filled with 0s except for a 1
# at index of the current letter, e.g. ``"b" = <0 1 0 0 0 ...>``.
#
# To make a word we join a bunch of those into a 2D matrix
# ``<line_length x 1 x n_letters>``.
#
# That extra 1 dimension is because PyTorch assumes everything is in
# batches - we're just using a batch size of 1 here.
#

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

    
class LastNames(torch.utils.data.Dataset):
    def __init__(self, split, N=None):
        super().__init__()
        
        self.split = split

        if split == "train":
            self.categories = ['French','Dutch']
        elif split == "val":
            self.categories = ['French','Dutch']
        elif split == "ood":
            self.categories = ['Chinese', 'Japanese', 'Korean']
        
        self.K = len(self.categories)
        self.N = 1000
        if N is not None:
            self.N = min(N, 1000)
    
    def __len__(self):
        return self.N
        
    def __getitem__(self, i):
        target = np.random.choice(self.K)
        category = self.categories[target]
        line = randomChoice(category_lines[category])
        
#         target = target % 5
        target = torch.Tensor([ target % 2 ] )
        line = lineToTensor(line)
        return line, target
    
    def TensorToLine(self, line):
        line = line.detach().cpu().numpy()
        line = np.argmax(line,axis=-1)
        line_str = ''
        for idx in line[:,0]:
            line_str += line_str.join(all_letters[int(idx)])
        return line_str
    
    def TargetToCategory(self, target):
        return self.categories[target]