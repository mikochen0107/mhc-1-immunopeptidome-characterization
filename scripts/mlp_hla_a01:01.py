import os
import sys
sys.path.append("..")  # add top folder to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_auc_score
import torch
import impepdom

model = impepdom.models.MultilayerPerceptron(num_hidden_layers=2, hidden_layer_size=100)
dataset = impepdom.PeptideDataset(
    hla_allele='HLA-A01:01',
    padding='flurry',
    toy=True)

folder, baseline_metrics, _ = impepdom.run_experiment(
    model,
    dataset,
    train_fold_idx=[1, 2, 3],
    val_fold_idx=[0],
    learning_rate=2e-3,
    num_epochs=5,
    batch_size=32)

trained_model, train_history = impepdom.load_trained_model(model, folder)
impepdom.plot_train_history(train_history, baseline_metrics)