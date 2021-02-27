'''
Preleminary Functions
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import cvxpy as cp
import warnings
import time
from numba import jit


warnings.filterwarnings("ignore", category=DeprecationWarning)
from linear_models import LogisticRegressor, RidgeRegressor
from kernel_models import KernelRidgeRegressor, KernelSVM

def write_csv(ids, labels, filename):
    """
    inputs:
        - ids: list of ids, should be an increasing list of integers
        - labels: list of corresponding labels, either 0 or 1
        - file: string containing the name that should be given to the submission file
    """
    df = pd.DataFrame({"Id": ids, "Bound": labels})
    df["Bound"] = df["Bound"].replace([-1], 0)
    df.to_csv(filename, sep=',', index=False)

def run_model(model_name, data_folder = 'data', prop_test = 0.05, kernel=None, sequence = False):
    dim = 100
    Nb_samples = 2000
    lamb = 0.5
    sigma = 1.2
    k=[4,5,6]

    if model_name == 'logreg':
        params = None
        model = LogisticRegressor()
    elif model_name == 'rr':
        params = {'lamb': np.linspace(0.001, 0.1, 20)}
        model = GridSearchCV(RidgeRegressor(), params)
    elif model_name == 'krr':
        params = {'lamb': np.linspace(0.1, 2, 2), 'sigma': np.linspace(0.5, 2, 20), 'kernel': ['gaussian']}
        model = GridSearchCV(KernelRidgeRegressor(), params)
    elif model_name == 'ksvm' and kernel == 'gaussian':
        params = {'lamb': np.logspace(-10., -7., 4), 'sigma': np.logspace(-1., 2., 4), 'kernel': ['gaussian']}
        model = GridSearchCV(KernelSVM(), params)
    elif model_name == 'ksvm' and kernel == 'spectrum':
        params = None
        model = KernelSVM(lamb = lamb, k=k, kernel='spectrum')
    else:
        print('model not defined')

    all_y_eval = []

    np.random.seed(1)
    for name in [0, 1, 2]:
        X = pd.read_csv(f'{data_folder}/Xtr{name}_mat100.csv', sep = ' ', index_col=False, header=None).to_numpy()
        mean, std = X.mean(axis=0), X.std(axis=0)
        X = (X - mean)/std
        y = pd.read_csv(f'{data_folder}/Ytr{name}.csv')

        if sequence == True:
            df = pd.read_csv(f'{data_folder}/Xtr{name}.csv')
            X = np.array(df['seq'])
            y = pd.read_csv(f'{data_folder}/Ytr{name}.csv')

        y = y["Bound"].to_numpy()

        if kernel is not None:
            y[y==0] = -1

        tr_indices = np.random.choice(Nb_samples, size=int((1-prop_test)*Nb_samples), replace=False)
        te_indices = [idx for idx in range(Nb_samples) if idx not in tr_indices]

        X_tr = X[tr_indices]
        X_te = X[te_indices]

        y_tr = y[tr_indices]
        y_te = y[te_indices]

        assert X_tr.shape[0] + X_te.shape[0] == X.shape[0]
        assert y_tr.shape[0] + y_te.shape[0] == y.shape[0]

        # Fitting
        model.fit(X_tr, y_tr)

        if params != None:
            print(model.best_params_)


        print(f"Accuracy on train set {name}: {model.score(X_tr, y_tr):.2f}")
        print(f"Accuracy on test set {name} : {model.score(X_te, y_te):.2f}")

        # Prediction on the new set
        X_eval = pd.read_csv(f'{data_folder}/Xte{name}_mat100.csv', sep = ' ', index_col=False, header=None).to_numpy()
        X_eval = (X_eval - mean)/std
        if sequence == True:
            X_eval = np.array(pd.read_csv(f'{data_folder}/Xte{name}.csv')['seq'])

        y_eval = model.predict(X_eval)
        all_y_eval.append(y_eval)

    all_y_eval = np.hstack(all_y_eval).reshape(-1)
