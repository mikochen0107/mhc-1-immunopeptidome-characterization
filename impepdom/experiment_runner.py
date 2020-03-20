import os
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import impepdom


STORE_PATH = '../store'

def run_experiment(
    model, hla_allele, train_fold_idx, val_fold_idx=None, padding='after2', toy=False,
    criterion=None, optimizer=None, scheduler=None,
    batch_size=64, num_epochs=25, learning_rate=1e-3, show_output=True
):
    '''
    Run a neural network training on specified train and validation set, with parameters.

    Parameters
    ----------
    model: nn.Module
        Defined neural network

    hla_allele: string
        Name of the folder corresponding to MHC I allele. Input example: 'HLA-A01:01'

    train_fold_idx: list
        List of number (from 0 to 4) to specify training folds

    val_fold_idx: list, optional
        List of number (from 0 to 4) to specify validation folds

    padding:
    
    toy: bool
        Load partial dataset

    criterion: nn.Loss, optional
    scheduler: torch.optim.lr_scheduler, optional
    batch_size: int, optional
    num_epochs: int, optional
    learning_rate: float, optional

    show_output: bool
        Show output of training process (everything will be saved anyway)

    Returns
    ----------
    save_folder: string
        Relative path where the cache is located
    '''

    # obtain torch dataloader for batch learning
    dataset = impepdom.PeptideDataset(hla_allele, padding=padding, toy=toy)
    peploader = {}
    peploader['train'] = dataset.get_peptide_dataloader(fold_idx=train_fold_idx, batch_size=batch_size)
    peploader['val'] = dataset.get_peptide_dataloader(fold_idx=val_fold_idx, batch_size=batch_size)

    # set up optimization criterion, optimization algorithm, and learning rate decay
    if criterion == None:
        criterion = nn.BCELoss()
    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler == None:
        steps = 10
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        # lr_decay_step = 5
        # decay_factor = 0.9
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=decay_factor)

    # collect baseline metrics
    baseline_metrics = get_baseline_metrics(dataset, train_fold_idx, val_fold_idx)

    # train the model, collect data
    need_validation = False if val_fold_idx is None else True
    model, train_history = impepdom.train_nn(
        model=model,
        peploader=peploader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        validation=need_validation)

    # save model
    save_folder = get_save_path(model, hla_allele, train_fold_idx)
    torch.save(model.state_dict(), os.path.join(save_folder, 'torch_model'))
    
    # save training history
    pickle_dump(train_history, save_folder, 'train_history')

    # save validation predictions
    val = {}
    data, val['target'] = dataset.get_fold(val_fold_idx)
    val['pred'] = model(torch.tensor(data).float())
    pickle_dump(val, save_folder, 'validation_' + list_to_str(val_fold_idx))

    return save_folder, baseline_metrics

def plot_train_history(train_history, baseline_metrics=None, metrics=['loss', 'acc', 'auc']):
    '''
    Plot metrics to observe training (vs validation) progress.

    Parameters
    ----------
    train_history: dict
        Dictionary containing training (and validation) metric logs over epochs
    baseline_metrics: dict, optional
        Dictionary containing reference metrics (e.g., if we guess everything as non-binding)
    '''

    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False

    num_epochs = len(train_history['train']['loss'])
    val_exists = len(train_history['val']['loss']) > 0 

    plt.figure(figsize=(16, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)

        plt.plot(range(num_epochs), train_history['train'][metric], color='skyblue', label='train')
        if val_exists:
            plt.plot(range(num_epochs), train_history['val'][metric], color='darkorange', label='val')
        if baseline_metrics and metric != 'loss':
            plt.axhline(y=baseline_metrics['train'][metric] + 1e-3, color='skyblue', alpha=0.7, linestyle='--', label='train base')
            if 'acc' in baseline_metrics['val']:
                plt.axhline(y=baseline_metrics['val'][metric] - 1e-3, color='darkorange', alpha=0.7, linestyle='--', label='val base')

        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend()

    plt.show()

def get_baseline_metrics(dataset, train_fold_idx, val_fold_idx):
    baseline_metrics = {
        'train': {},
        'val': {}
    }

    _, train_targets = dataset.get_fold(fold_idx=train_fold_idx)
    train_zeros = np.zeros(train_targets.shape)
    if val_fold_idx:
        _, val_targets = dataset.get_fold(fold_idx=val_fold_idx)
        val_zeros = np.zeros(val_targets.shape)

    baseline_metrics['train']['acc'] = np.sum((train_targets == train_zeros)) / len(train_zeros)
    baseline_metrics['train']['auc'] = roc_auc_score(train_targets, train_zeros)
    if val_fold_idx:
        baseline_metrics['val']['acc'] = np.sum((val_targets == val_zeros)) / len(val_zeros)
        baseline_metrics['val']['auc'] = roc_auc_score(val_targets, val_zeros)

    return baseline_metrics


def load_trained_model(model, save_folder):
    '''
    Load from information from /store 
    
    Parameters
    ----------
    model: nn.Module
        Initialized class of the model to be loaded
    save_folder: string
        Path to folder storing trained model

    Returns
    ----------
    model: string
        Return trained torch model
    
    train_history: dict
        Dictionary of training history
    '''

    # fetch trained torch model
    model_path = os.path.join(save_folder, 'torch_model')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # set to inference mode

    # fetch training history
    train_history_path = os.path.join(save_folder, 'train_history')
    infile = open(train_history_path,'rb')
    train_history = pickle.load(infile)
    infile.close()

    return model, train_history

def get_save_path(model, hla_allele, train_fold_idx):
    '''
    Saves model trained state and history of training and validation metrics.

    Parameters
    ----------
    hla_allele: string
        Name of the MHC allele used for training
    
    train_fold_idx: list
        Indices of folds used for training

    Returns
    ----------
    path_to_folder: string
        Relative path to folder storing model information
    '''

    name = model.get_my_name()
    allele = hla_allele[4:].lower()  # crop the "hla-" part since they all share the same 
    train_folds = list_to_str(train_fold_idx)

    # storage format is model_name - hla_allele - fold indices - date, time of model save
    dt = datetime.now().strftime('%y%m%d%H%M%S')
    folder = '{0}_{1}_{2}_'.format(name, allele, train_folds) + dt
    path_to_folder = os.path.join(STORE_PATH, folder)
    os.makedirs(path_to_folder, exist_ok=True)

    return path_to_folder

def list_to_str(ls):
    _str = ''.join(sorted([str(c) for c in ls]))
    return _str

def pickle_dump(data, save_folder, filename):
    path = os.path.join(save_folder, filename)
    out = open(path, 'wb')
    pickle.dump(data, out)