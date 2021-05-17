import time
import copy
import torch
import numpy as np
from tqdm.autonotebook import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

def train_model(model, dataset_class, dist_fam, optimizer, scheduler, device, num_epochs=25, batch_size=4, log_every=100):
    """
    Trains a model on datatsets['train'] using criterion(model(inputs), labels) as the loss.
    Returns the model with lowest loss on datasets['val']
    Optimizes using optimizer whose learning rate is controlled by scheduler.
    Puts model and inputs on device.
    Trains for num_epochs passes through both datasets.
    
    Writes tensorboard info to ./runs/ if given
    """
    writer = None
    writer = SummaryWriter()
        
    model = model.to(device)
    
    since = time.time()

    datasets = {x: dataset_class(x) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    
    tr_loss = np.nan
    val_loss = np.nan
    
    tr_metric = np.nan
    val_metric = np.nan
    
    n_tr_batches_seen = 0
    
    with tqdm(total=num_epochs, position=0) as pbar:
        pbar2 = tqdm(total=dataset_sizes['train'], position=1)
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_metric = 0.0

                running_tr_loss = 0.0 # used for logging
                running_tr_metric = 0.0 # used for logging

                running_n = 0
                
                # Iterate over data.
                pbar2.refresh()
                pbar2.reset(total=dataset_sizes[phase])
                for inputs, labels in dataloaders[phase]:
                    if phase == 'train':
                        n_tr_batches_seen += 1
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = dist_fam.loss(outputs, labels).mean()
                        metric_name, metric = dist_fam.metric(dist_fam.output(outputs), labels)
                        metric = metric.mean()
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.shape[0]
                    running_metric +=  metric.item()*inputs.shape[0]
                    
                    if phase =='train':
                        running_n += inputs.shape[0]
                        running_tr_loss += loss.item() * inputs.shape[0]
                        running_tr_metric +=  metric.item()*inputs.shape[0]
                        
                        if n_tr_batches_seen % log_every == 0:
                            mean_loss = running_tr_loss / running_n
                            mean_metric = running_tr_metric / running_n
                            
                            writer.add_scalar('loss/train', mean_loss, n_tr_batches_seen)
                            writer.add_scalar('metric/train', mean_metric, n_tr_batches_seen)
                            
                            running_tr_loss = 0.
                            running_tr_metric = 0.
                            running_n = 0
                    
                    pbar2.set_postfix(split=phase, batch_loss=loss.item(), batch_metric=metric.item())
                    pbar2.update(inputs.shape[0])
                    

                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_metric = running_metric / dataset_sizes[phase]
                
                if phase == 'train':
                    tr_loss = epoch_loss
                    tr_metric = epoch_metric
                    scheduler.step()
                    
                if phase == 'val':
                    val_loss = epoch_loss
                    val_metric = epoch_metric
                    writer.add_scalar('loss/val', val_loss, n_tr_batches_seen)
                    writer.add_scalar('metric/val', val_metric, n_tr_batches_seen)

                pbar.set_postfix(tr_loss=tr_loss, val_loss=val_loss, tr_metric=tr_metric, val_metric=val_metric)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
            pbar.update(1)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    
    
    writer.flush()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_ensemble(K, model_init_fn, dataset_class, criterion, opt_class, opt_kwargs, sched_class, sched_kwargs, device, **kwargs):
    """
    Sequentially trains K models initialized by model_init_fn (assumed to have some randomization built in)
    
    The remaining arguments are the same as train_model
    """
    models = []
    for j in range(K):
        print("Training model %d of %d" % (j+1,  K))
        model = model_init_fn()
        optimizer = opt_class(model.parameters(), **opt_kwargs)
        scheduler = sched_class(optimizer, **sched_kwargs)
        train_model(model, dataset_class, criterion, optimizer, scheduler, device, **kwargs)
        
        models.append(model)
    
    return models


        