import os
import time
from datetime import datetime

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_nn(model, peploader, criterion, optimizer, scheduler=None, num_epochs=25, learning_rate=1e-2, show_output=True):
    '''
    Train neural network using gradient descent and backprop

    Parameters
    ----------
    model: nn.Module
        Defined neural network
    peploader: dict
        Dictionary of torch.utils.data.DataLoader for train, val
    
    Returns
    ----------
    model: nn.Module
        Trained neural network
    train_history: dict
        Dictionary of defined metrics for train and val sets for each epoch
    '''

    since = time.time()
    train_history = init_train_hist()
    
    for e in range(num_epochs):
        print('epoch {}/{} at {:.4f}'.format(e, num_epochs - 1, time.time() - since))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval(()  # set model to evaluation mode

            count = 0  # just use length of data
            running_loss = 0.0
            running_acc = 0.0

            for item in peploader[phase]:
                pep = item['peptide']
                target = item['target'].view(-1, 1)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(pep.float())
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, target)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * pep.size[0]
                running_acc += torch.sum(preds == target) # change to AUC
                count += 1
            
            # add epoch metrics to training history
            epoch_loss = running_loss / count
            epoch_acc = running_acc / count
            train_history[phase]['loss'].append(epoch_loss)
            train_history[phase]['acc'].append()
            
            stdout.flush()  # before printing
            print('{} loss: {:.4f} accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()  # empty line

    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('best validation acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, train_history

def init_train_hist():
    metrics = ['loss', 'acc', 'auc', 'auc0.1', 'pcc']
    train_history = {
        'train': {},
        'val': {}
    }

    for metric in metrics:
        train_history['train'][metric] = []
        train_history['val'][metric] = []

    return train_history