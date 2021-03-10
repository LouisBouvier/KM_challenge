import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import cvxpy as cp
import warnings
import time
from itertools import product
from numba import jit

from utils import run_model, write_csv


warnings.filterwarnings("ignore", category=DeprecationWarning)

data_folder = 'data' # 'machine-learning-with-kernel-methods-2021'

from kernels import Mismatch_kernel

K = []
k=9
m=4
for name in [0, 1, 2]:
    X    = np.array(pd.read_csv(f'{data_folder}/Xtr{name}.csv')['seq'])
    X_ev = np.array(pd.read_csv(f'{data_folder}/Xte{name}.csv')['seq'])

    t0 = time.time()
    K_tr = Mismatch_kernel(X, X, k=k, m=m)
    np.save('results/dataset_{}_mismatch_K_tr_{}_{}.npy'.format(name,k,m), K_tr)
    print(f"Time to compute train kernel : {time.time() - t0}")
    K_te = Mismatch_kernel(X, X_ev, k=k, m=m)
    np.save('results/dataset_{}_mismatch_K_te_{}_{}.npy'.format(name,k,m), K_te)


    K.append({"train": K_tr, "eval": K_te})
