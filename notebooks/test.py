import sys
sys.path.append("..")  # add top folder to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import impepdom

hla_a02_01 = impepdom.PeptideDataset('HLA-A02:01', padding='after2')
hla_a02_01.basic_dataviz()