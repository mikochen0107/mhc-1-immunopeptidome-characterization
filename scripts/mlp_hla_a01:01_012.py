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


model = impepdom.MultilayerPerceptron()
impepdom.run_experiment(model, learning_rate=3)