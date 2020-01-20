import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from data_utils import getHeader, findCorrelation, generateReport
from distance_calculator import calculateDistance

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

distance_parameter = ['Frobenius Norm', 'Energy Distance', 'L2 Norm', 'Wasserstein Distance', 'Frechet Inception Distance',
                      'Students T-test', 'KS Test', 'Shapiro Wil Test', 'Anderson Darling Test']

threshold = 0.5
real_train_file = 'x_train.npy'
syn_train_file = '0.npy'
header_file = "x_headers.txt"
lst = getHeader(header_file)
# Real Data and Synthetic Data
corr_real, corr_syn, chng_lst = findCorrelation(real_train_file, syn_train_file, lst)

# Deleted columns list
del_list = [x for x in lst if x not in chng_lst]

# Distance Calculator
distances = calculateDistance(corr_real, corr_syn, distance_parameter)

# Generate Report
generateReport(distances)
