import torch
import numpy as np
import time
import tqdm
from .data_transforms import TransformedDataset



def test_unc_model(model, dataset, device, batch_size=1, **forward_kwargs):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    
    dataset_size = len(dataset)
    
    metrics = []
    unc_list = []
    times_per_item = []
    metric_name = "metric"
    
    # Iterate over data.
    for inputs, labels in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        # forward
        # track history if only in train
        start = time.time()
        outputs, uncs = model(inputs, **forward_kwargs)
        end = time.time()
                
        times_per_item.append( (end - start)/inputs.shape[0] )
        
        metric_name, metric = model.dist_fam.metric(outputs, labels)
        
        metrics.append( metric.cpu().detach().numpy() )
        unc_list.append( uncs.cpu().detach().numpy() )

    results = {
        "metric_name": metric_name,
        "metrics": np.concatenate(metrics),
        "uncs": np.concatenate(unc_list),
        "time_per_item": np.mean(times_per_item),
        "time_per_item_std": np.std(times_per_item) / np.sqrt(dataset_size),
    }
        
    return results


def process_datasets(dataset_class, dataset_args, unc_model, device, test_fn=test_unc_model, N=1000, **forward_kwargs):
    """
    calls test_fn(unc_model, dataset) for every dataset in datasets
    """
    results = {}
    for name in dataset_args:
        print("Testing", name)
        print(N)
        dataset = dataset_class(name,N=N)
        results[name] = test_fn(unc_model, dataset, device, **forward_kwargs)
    
    return results

def transform_sweep(dataset, transforms, unc_model, device, test_fn=test_unc_model, **forward_kwargs):
    """
    calls test_fn(unc_model, dataset, device, **forward_kwargs) on dataset as well as 
    transformed versions of that dataset, augmenting the passed in dataset with each 
    transform in transforms (a dictionary with entries: "transform_name": transform
    """
    results = {}
    
    print("Testing original dataset")
    results["original"] = test_fn(unc_model, dataset, device, **forward_kwargs)
    for name, transform in transforms.items():
        print("Testing", name)
        dataset = TransformedDataset(dataset, transform)
        results[name] = test_fn(unc_model, dataset, device, **forward_kwargs)

    return results


