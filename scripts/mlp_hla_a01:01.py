import os
import sys
sys.path.append("..")  # add top folder to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_auc_score
import impepdom
import torch


model = impepdom.MultilayerPerceptron(num_hidden_layers=3)
save_folder, baseline_metrics = impepdom.run_experiment(
    model,
    hla_allele='HLA-A01:01',
    train_fold_idx=[0, 1, 2],
    val_fold_idx=[3],
    learning_rate=8e-2,
    num_epochs=25,
    toy=False)

trained_model, train_history = impepdom.load_trained_model(model, save_folder)
impepdom.plot_train_history(train_history, baseline_metrics)