import os
import time
import copy

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import roc_auc_score
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
            
            # add epoch metrics to training history
            y_actual = np.vstack(y_actual)
            y_pred = np.vstack(y_pred)
            y_proba = np.vstack(y_proba)

            # calculate metrics for the model at current epoch
            epoch_loss = running_loss / count
            epoch_acc = np.sum(y_actual == y_pred) / count
            epoch_f1 = impepdom.metrics.ppv(y_actual, y_pred) 

            epoch_auc = roc_auc_score(y_actual, y_proba)
            epoch_auc_01 = impepdom.metrics.auc_01(y_actual, y_proba)

            epoch_ppv = impepdom.metrics.ppv(y_actual, y_proba) 
            epoch_ppv_100 = impepdom.metrics.ppv_100(y_actual, y_proba)

            # save calculated metrics to the training history
            train_history[phase]['loss'].append(epoch_loss)
            train_history[phase]['acc'].append(epoch_acc)
            train_history[phase]['f1'].append(epoch_f1)

            train_history[phase]['auc'].append(epoch_auc)
            train_history[phase]['auc_01'].append(epoch_auc_01)

            train_history[phase]['ppv'].append(epoch_ppv)
            train_history[phase]['ppv_100'].append(epoch_ppv_100)
            
            if show_output:
                print('{} loss: {:.4f} accuracy: {:.4f} auc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_auc))

            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
            elif not validation:  # just take the latest best result
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        if show_output:
            print()  # empty line

    time_elapsed = time.time() - since
    if show_output:
        print('training completed in {:.0f} m {:.4f} s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('best validation auc: {:.4f}'.format(best_auc))

    model.load_state_dict(best_model_wts)
    return model, train_history

def init_train_hist():
    '''
    Get a placeholder for a training logs storage.

    Returns
    ----------
    train_history: dict
        Dictionary to contain training (and validation) metric logs over epochs
    '''

    metrics = ['loss', 'acc', 'f1', 'auc', 'auc_01', 'ppv', 'ppv_100']
    train_history = {
        'train': {},
        'val': {}
    }

    for metric in metrics:
        train_history['train'][metric] = []
        train_history['val'][metric] = []

    return train_history