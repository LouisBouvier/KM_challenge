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


def init_model(model_name, default_params, kernel=None, precomputed_kernel=None, use_grid_search=False):
    """
    Initializes a model depending on the parameters specified.
    """
    if model_name == 'logreg':
        params = None
        model = LogisticRegressor()
    elif model_name == 'rr':
        params = {'lamb': np.linspace(0.001, 0.1, 20)}
        model = RidgeRegressor()
    elif model_name == 'krr':
        if precomputed_kernel is not None:
            raise NotImplementedError("Using a precomputed kernel is only available for the Kernel SVM.")
        params = {'lamb': np.linspace(0.1, 2, 2), 'sigma': np.linspace(0.5, 2, 20), 'kernel': ['gaussian']}
        model = KernelRidgeRegressor(lamb=default_params['lamb'], sigma=default_params['sigma'], kernel='gaussian')
    elif model_name == 'ksvm':
        if precomputed_kernel is not None:
            params = {'lamb': np.logspace(-10., -7., 4)} # We don't have other values because they have already been used to compute the kernel
            model = KernelSVM(lamb=default_params['lamb'], precomputed_kernel=precomputed_kernel)
        elif kernel == 'gaussian':
            params = {'lamb': np.logspace(-10., -7., 4), 'sigma': np.logspace(-1., 2., 4), 'kernel': ['gaussian']}
            model = KernelSVM(lamb=default_params['lamb'], sigma=default_params['sigma'], kernel='gaussian')
        elif kernel == 'spectrum':
            params = None
            model = KernelSVM(lamb=default_params['lamb'], k=default_params['k'][0], kernel='spectrum')
        elif kernel == 'substring':
            params = None
            model = KernelSVM(lamb=default_params['lamb'], k=default_params['k'][0], kernel='substring')
    else:
        print('model not defined')

    if use_grid_search and params is not None:
        model = GridSearchCV(model, params)

    return model


def run_model(model_name,
              data_folder = 'data',
              prop_test = 0.05,
              kernel=None,
              kernel_savefiles=None,
              K=None,
              sequence = False,
              use_grid_search = False):
    """
    inputs:
        - model_name (str): name of the model used for classification
        - data_folder (str): relative path to the data folder
        - prop_test (float): proportion of examples to use for testing
        - kernel (str): name of the kernel to use
        - kernel_savefiles (list of dict of str): contains paths to kernel saved as numpy matrices
        - K (list of dict of arrays): kernels already computed for training and evaluation for each dataset.
        - sequence (bool): if True, use the data under the sequence form. If False, use precomputed representations.
        - use_grid_search (bool): set to True if you want to use GridSearchCV

    output:
        - array with the predictions over the whole evaluation set.
    """
    dim = 100
    Nb_samples = 2000
    default_params = {'lamb': 0.5, 'sigma': 1.2, 'k': [4, 5, 6]}

    all_y_eval = []

    np.random.seed(1)
    for name in [0, 1, 2]:
        # Load training / testing sets
        X = pd.read_csv(f'{data_folder}/Xtr{name}_mat100.csv', sep = ' ', index_col=False, header=None).to_numpy()
        mean, std = X.mean(axis=0), X.std(axis=0)
        X = (X - mean)/std
        y = pd.read_csv(f'{data_folder}/Ytr{name}.csv')

        if sequence:
            df = pd.read_csv(f'{data_folder}/Xtr{name}.csv')
            X = np.array(df['seq'])
            y = pd.read_csv(f'{data_folder}/Ytr{name}.csv')

        # Load evaluation set
        X_eval = pd.read_csv(f'{data_folder}/Xte{name}_mat100.csv', sep = ' ', index_col=False, header=None).to_numpy()
        X_eval = (X_eval - mean)/std

        if sequence:
            df_eval = pd.read_csv(f'{data_folder}/Xte{name}.csv')
            X_eval = np.array(df_eval['seq'])

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


        if sequence and kernel_savefiles is not None:
            precomputed_kernel = load_precomputed_kernel(df, df_eval,
                                                         kernel_filename_train=kernel_savefiles[name]['train'],
                                                         kernel_filename_eval=kernel_savefiles[name]['eval'])

        elif sequence and K is not None:
            precomputed_kernel = load_precomputed_kernel(df, df_eval, K_tr=K[name]['train'], K_ev=K[name]['eval'])

        model = init_model(model_name,
                           default_params,
                           kernel=kernel,
                           precomputed_kernel=precomputed_kernel,
                           use_grid_search=use_grid_search)

        # Fitting
        model.fit(X_tr, y_tr)

        if use_grid_search:
            print(model.best_params_)

        print(f"Accuracy on train set {name}: {model.score(X_tr, y_tr):.2f}")
        print(f"Accuracy on test set {name} : {model.score(X_te, y_te):.2f}")

        # Prediction on the new set
        y_eval = model.predict(X_eval)
        all_y_eval.append(y_eval)

    all_y_eval = np.hstack(all_y_eval).reshape(-1)

    return all_y_eval


def load_precomputed_kernel(df_train, df_eval,
                            kernel_filename_train=None, kernel_filename_eval=None,
                            K_tr=None, K_ev=None,
                            normalize=False, make_it_positive=False):
    """
    Create a function that will compute the kernel between datapoints by finding the correct indices and using a precomputed kernel to return the solution.

    """
    if kernel_filename_train is not None and kernel_filename_eval is not None:
        with open(kernel_filename_train, "rb") as f:
            K_tr = np.load(f)
        with open(kernel_filename_eval, "rb") as f:
            K_ev = np.load(f)
    elif K_tr is None and K_ev is None:
        raise ValueError("You need to specify a method for loading a preexisting kernel.")

    def precomputed_kernel(X1, X2, **args):
        K = np.zeros((len(X1), len(X2)))

        idx1 = []
        for x in X1: # Needed to get elements in the right order
            idx1.append(df_train[df_train['seq'] == x]['Id'].iloc[0] % 2000)

        if sum(df_train['seq'].isin(X2)) >= len(X2): # Check if all elements are in the training set
            idx2 = []
            for x in X2: # Needed to get elements in the right order
                idx2.append(df_train[df_train['seq'] == x]['Id'].iloc[0] % 2000)
            # Extract submatrix using correct indices
            K = K_tr[np.ix_(idx1, idx2)]

        elif sum(df_eval['seq'].isin(X2)) >= len(X2): # Check if all elements are in the evaluation set
            idx2 = []
            for x in X2: # Needed to get elements in the right order
                idx2.append(df_eval[df_eval['seq'] == x]['Id'].iloc[0] % 1000)
            # Extract submatrix using correct indices
            K = K_ev[np.ix_(idx1, idx2)]

        if make_it_positive: # This is to have a positive matrix 'à la bled'
            K += 1e-8
        # Divide all elements by row and column values. Not encouraged.
        if normalize:
            diag = np.copy(np.diag(K))
            K = K / diag[:, None]
            K = K / diag
        return K


    return precomputed_kernel
