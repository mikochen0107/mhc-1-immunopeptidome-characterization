import sys
sys.path.append("..")  # add top folder to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import impepdom


hla_a01_01 = impepdom.PeptideDataset('HLA-A01:01', padding='after2', toy=True)
peploader = hla_a01_01.get_peptide_dataloader(fold_idx=[0, 1, 2])
mlp = impepdom.MultilayerPerceptron()
impepdom.train_nn(mlp, peploader)