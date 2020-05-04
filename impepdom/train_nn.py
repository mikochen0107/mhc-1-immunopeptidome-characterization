import os
import time
import copy

import torch
import torch.nn as nn
import numpy as np

import impepdom.metrics

def train_nn(model, peploader, criterion, optimizer, scheduler=None, num_epochs=25, learning_rate=1e-2, validation=True, show_output=True):
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
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    
    # enable training on whole dataset after all validations
    phases = ['train']
    if validation == True:
        phases.append('val')

    for e in range(num_epochs):
        if show_output:
            print('epoch {}/{} started at {:.4f} s'.format(e + 1, num_epochs, time.time() - since))
        for phase in phases:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluation mode
            
            count = 0  # just use length of data
            running_loss = 0.0

            # storage for targets and predictions from batches
            y_actual = []
            y_pred = []
            y_proba = []

            for item in peploader[phase]:
                pep = item['peptide']
                target = item['target'].view(-1, 1)
                optimizer.zero_grad()
               
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(pep.float())                
                    _pos = torch.ones(output.size())
                    _neg = torch.zeros(output.size())
                    preds = torch.where(output > 0.5, _pos, _neg)
                    proba = output.detach()

                    y_actual.append(target.numpy())
                    y_pred.append(preds.numpy())
                    y_proba.append(proba.numpy())

                    loss = criterion(output, target)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # decay learning rate every few epochs
                
                running_loss += loss.item() * pep.size(0)
                count += pep.size(0)
            
            # add epoch labels and predictions to training history
            train_history[phase]['out']['actual'].append(np.vstack(y_actual))
            train_history[phase]['out']['pred'].append(np.vstack(y_pred))
            train_history[phase]['out']['proba'].append(np.vstack(y_proba))

            # calculate metrics for the model at current epoch
            train_history[phase]['metrics']['loss'].append(running_loss / count) 
            train_history[phase]['metrics'] = impepdom.metrics.calculate_metrics(train_history)[phase]

            epoch_loss = train_history[phase]['metrics']['loss'][-1]
            epoch_acc = train_history[phase]['metrics']['acc'][-1]
            epoch_auc = train_history[phase]['metrics']['auc'][-1]
            
            if show_output:
                print('{} loss: {:.4f} acc: {:.4f} auc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_auc))

            # select best model
            if (phase == 'val' or not validation) and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        if show_output:
            print()  # empty line

    time_elapsed = time.time() - since
    if show_output:
        print('training completed in {:.0f} m {:.4f} s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('best {} auc: {:.4f}'.format('validation' if validation else 'training', best_auc))

    model.load_state_dict(best_model_wts)
    return model, train_history

def init_train_hist():
    '''
    Get a placeholder for a training logs storage.

    Returns
    ----------
    train_history: dict
        Dictionary to contain training (and validation) output and metric logs over epochs
    '''

    metrics = ['loss'] + impepdom.metrics.METRICS
    train_history = {
        'train': {'metrics': {}, 'out': {}},
        'val': {'metrics': {}, 'out': {}}
    }

    for metric in metrics:
        train_history['train']['metrics'][metric] = []
        train_history['val']['metrics'][metric] = []

    out_types = ['actual', 'proba', 'pred'] 

    # initialize dictionaries for model outputs
    for out_type in out_types:
        train_history['train']['out'][out_type] = []
        train_history['val']['out'][out_type] = []

    return train_history


'''
train_history['train']['metrics']['pcc']
train_history['train']['out']['pred'], ['proba'], ['actual']
train_history['train']
'''