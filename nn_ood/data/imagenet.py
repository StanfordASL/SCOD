from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import os

# TODO: define class_name to imagenet1000 index
train_classes = {
    'SaintBernard': 0,
    'bagel': 1,
    'jeep': 2,
}

# ood_1_classes = {
#     'shiitake': 0,
#     'morel': 1,
# }

# ood_2_classes = {
#     'agaric': 0,
#     'bolete': 1,
#     'earthstar': 2,
# }


ood_classes = {
    'hen-of-the-woods': 0,
#     'stinkhorn': 1,
#     'coralfungus': 2,
    'giraffe': 1,
    'pizza': 2,
    'iguana': 1,
    'fryingpan': 2
}

class ImageNetData(Subset):
    def __init__(self, split, N=None):
        root = "/home/apoorva/datasets/part-of-imagenet/partial_imagenet"
        extensions = ("jpg",)
        
        def is_valid_file(path):
            if path[-3:] != "jpg":
                return False
            try:
                pil_img = Image.open(path).convert('RGB')
            except:
                print("deleting", path)
                os.remove(path)
                return False
            return True

        def loader(path):
            pil_img = Image.open(path).convert('RGB')
            return pil_img

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        self.dataset = DatasetFolder(root, loader, 
#                                       extensions=extensions,
                                     is_valid_file=is_valid_file, 
                                     transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        
        if split == "train":
            class2idx = train_classes
        elif split == "val":
            class2idx = train_classes
        elif split == "ood":
            class2idx = ood_classes
#         elif split == "ood_1":
#             class2idx = ood_1_classes
#         elif split == "ood_2":
#             class2idx = ood_2_classes
#         elif split == "ood_3":
#             class2idx = ood_3_classes
            
        self.class2idx = class2idx
        
        self.valid_classes = np.argwhere([x in class2idx.keys() for x in self.dataset.classes])
        
                
        target_criterion = lambda x: x in self.valid_classes
        
        # filter out only 1s
        valid_idx = np.flatnonzero([t in self.valid_classes for t in self.dataset.targets])
        np.random.shuffle(valid_idx)
        
        if N is not None:
            valid_idx = valid_idx[:N]
        
        super().__init__(self.dataset, valid_idx)
        
        
    def __getitem__(self, i):
        input, target = super(ImageNetData, self).__getitem__(i)
        className = self.dataset.classes[target]
        target = self.class2idx[className]
        return input, target

    
