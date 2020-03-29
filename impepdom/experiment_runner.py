import os
import pickle
from itertools import permutations
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

import impepdom


def hyperparam_grid_search(
    model, dataset, fold_idx=[0, 1, 2, 3],
    max_epochs=15, batch_sizes=[32, 64, 128], learning_rates=[5e-4, 1e-3, 5e-3],
    optimizer=None, scheduler=None, sort_by='mean_auc_01'
):
    '''
    Parameters
    ----------
    model: nn.Module
        Defined neural network

    dataset: impepdom.PeptideDataset
        Initialized peptide dataset for MHC I

    fold_idx: list
        List of number (from 0 to 4) to specify k-folds for cross-validation

    max_epochs: int, optional
        Epoch at which to stop training. Results of all intermediary epochs will be saved

    batch_sizes: list, optional

    learning_rates: list, optional
    criterion: nn.Loss, optional  # in the works
    scheduler: torch.optim.lr_scheduler, optional  # in the works

    sort_by: string, optional
        Sort results to present, desc_stat + metric. Examples: 'mean_acc', 'min_auc_01', 'max_ppv'

    Returns
    ---------- 
    results_store: list
        List of model results
    '''
    
    since = time.time()
    results_store = []  # to store model names and scores
    tot_experiments = len(batch_sizes) * len(learning_rates)
    experiment_count = 0
    padding = dataset.padding

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            experiment_count += 1
            print('running experiment {} out of {} at {:.4f} s'.format(experiment_count, tot_experiments, time.time() - since))
            metrics = impepdom.metrics.METRICS
            desc_stats = impepdom.metrics.DESC_STATS
            
            cross_eval = {}  # store history of validation metrics. Per metric: columns are epochs, rows are results on folds
            for metric in metrics:
                cross_eval[metric] = []
            
            which_model = None
            for val_fold_id in fold_idx:
                train_fold_idx = copy.copy(fold_idx)
                train_fold_idx.remove(val_fold_id)  # remove validation fold

                folder, _, config = run_experiment(
                    model,
                    dataset,
                    train_fold_idx=train_fold_idx,
                    val_fold_idx=[val_fold_id],
                    learning_rate=learning_rate,
                    num_epochs=max_epochs,
                    batch_size=batch_size,
                    scheduler=scheduler,
                    show_output=False,
                    which_model=which_model
                )

                _, train_history = impepdom.load_trained_model(model, folder)            
                for metric in metrics:
                    cross_eval[metric].append(train_history['val'][metric])  # get metric over epochs
                which_model = impepdom.store_manager.extract_which_model(folder)  # to keep in the same folder
            
            for epoch in range(max_epochs):
                res_obj = {
                    'model': folder[:folder.find('/')],
                    'padding': padding,
                    'batch_size': batch_size,
                    'num_epochs': epoch + 1,
                    'learning_rate': learning_rate,
                    'optimizer': config['optimizer'],
                    'scheduler': config['scheduler']
                }

                for metric in metrics:
                    cross_eval_metric = np.vstack(cross_eval[metric])  # make into one numpy array
                    for desc_stat in desc_stats:
                        res_obj[desc_stat[0] + '_' + metric] = desc_stat[1](cross_eval_metric[:, epoch])

                results_store.append(res_obj)

    results_store.sort(key=(lambda model_res: model_res[sort_by]))
    impepdom.store_manager.update_hyperparams_store(results_store)

    time_elapsed = time.time() - since
    print('evaluation completed in {:.0f} m {:.4f} s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('best model for {0} is {1}'.format(sort_by, results_store[-1]))

    return results_store


def run_experiment(
    model, dataset, train_fold_idx, val_fold_idx=None,
    criterion=None, optimizer=None, scheduler=None,
    batch_size=64, num_epochs=25, learning_rate=1e-3, show_output=True,
    which_model=None
):
    '''
    Run a neural network training on specified train and validation set, with parameters.

    Parameters
    ----------
    model: nn.Module
        Defined neural network

    dataset: impepdom.PeptideDataset
        Initialized peptide dataset for MHC I

    train_fold_idx: list
        List of number (from 0 to 4) to specify training folds

    val_fold_idx: list, optional
        List of number (from 0 to 4) to specify validation folds

    criterion: nn.Loss, optional
    scheduler: torch.optim.lr_scheduler, optional
    batch_size: int, optional
    num_epochs: int, optional
    learning_rate: float, optional

    show_output: bool
        Show output of training process (everything will be saved anyway)

    which_model: string
        Attach results to the same model if we're just doing cross-validation

    Returns
    ----------
    folder: string
        Path inside STORE_PATH to folder storing training cache

    baseline_metrics: dict
        Dictionary of baseline metrics in train (and val) datasets

    config: dict
        Dictionary of configurations used in the training process
    '''
    
    need_validation = False if val_fold_idx is None else True

    # obtain torch dataloader for batch learning
    peploader = {}
    peploader['train'] = dataset.get_peptide_dataloader(fold_idx=train_fold_idx, batch_size=batch_size)
    if need_validation:
        peploader['val'] = dataset.get_peptide_dataloader(fold_idx=val_fold_idx, batch_size=batch_size)

    # set up optimization criterion, optimization algorithm, and learning rate decay
    if criterion == None:
        criterion = nn.BCELoss()
    if optimizer == None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler == None:
        steps = 25
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        # lr_decay_step = 5
        # decay_factor = 0.9
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=decay_factor)

    # collect baseline metrics
    baseline_metrics = get_baseline_metrics(dataset, train_fold_idx, val_fold_idx)

    # train the model, collect data
    model, train_history = impepdom.train_nn(
        model=model,
        peploader=peploader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        validation=need_validation,
        show_output=show_output
    )

    # save model
    folder = impepdom.store_manager.get_save_path(model, dataset.get_allele(), train_fold_idx, which_model=which_model)
    torch.save(model.state_dict(), os.path.join(impepdom.store_manager.STORE_PATH, folder, 'torch_model'))
    
    # save training history
    impepdom.store_manager.pickle_dump(train_history, folder, 'train_history')

    # save validation predictions
    if need_validation:
        val = {}
        data, val['target'] = dataset.get_fold(val_fold_idx)
        val['pred'] = model(torch.tensor(data).float())
        impepdom.store_manager.pickle_dump(val, folder, 'val_' + impepdom.store_manager.list_to_str(val_fold_idx))

    config = {
        # pass these parameters in a more obvious and descriptive way in the future
        'optimizer': str(optimizer)[:str(optimizer).find(' ')],
        'scheduler': 'CosineAnnealingLR'  # hard code for now
    }

    return folder, baseline_metrics, config

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
    metrics = impepdom.metrics.METRICS

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

def list_to_str(ls):
    _str = ''.join(sorted([str(c) for c in ls]))
    return _str

