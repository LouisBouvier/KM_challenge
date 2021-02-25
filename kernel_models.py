'''
Kernel Models:
- kernel Ridge Regression
- kernel SVM
'''

import numpy as np
import cvxpy as cp
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin

from kernels import Gaussian_kernel, Spectrum_kernel


class KernelRidgeRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self, lamb=1., sigma=1., kernel='gaussian'):
        """
        This class implements methods for fitting and predicting with a KernelRidgeRegressor used for classification
        (by thresholding the value regressed). Any kernel can be used.
        inputs:
        - lamb : the regularisation parameter
        - sigma : the parameter of the Gaussian kernel (if Gaussian kernel selected)
        - kernel : the kernel we consider
        """
        self.lamb = lamb
        self.sigma = sigma
        self.kernel = kernel
        if self.kernel == 'gaussian':
            self.kernel_ = partial(Gaussian_kernel, sig=sigma)
        else:
            raise NotImplementedError(f"Kernel {self.kernel} is not implemented yet")

    def fit(self, X, y):
        """
        inputs:
        - X (size: N_trxd): the points of the training set
        - y (size: N_trx1): the values of the classes
        """
        # We keep values of training in memory for prediction
        self.X_tr_ = np.copy(X)
        K = self.kernel_(X, X, sig=self.sigma)
        self.alpha_ = np.linalg.inv(K+self.lamb*X.shape[0]*np.eye(X.shape[0]))@y

        return self

    def predict(self, X):
        """
        inputs:
        - X (size N_texd): the points in R^d we want to classify
        output:
         - the predicted class for the associated y given the
        Linear Regression parameters
        """
        K_tr_te = self.kernel_(self.X_tr_, X, sig=self.sigma)

        return 2 * (self.alpha_.T@K_tr_te > 0).reshape(-1, ).astype("int") - 1

    def score(self, X, y):
        """
        inputs:
        - X (size N_texd): the points in R^d we want to classify
        - y (size N_tex1): the labels of the points
        """
        y_pred = self.predict(X)

        return np.sum(y_pred == y)/y.shape[0]

class KernelSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, lamb=1., sigma=1., k = 3, kernel='gaussian'):
        """
        This class implements methods for fitting and predicting with a KernelRidgeRegressor used for classification
        (by thresholding the value regressed). Any kernel can be used.
        inputs:
        - lamb : the regularisation parameter
        - sigma : the parameter of the Gaussian kernel (if Gaussian kernel selected)
        - kernel : the kernel we consider
        """
        self.lamb = lamb
        self.sigma = sigma
        self.k = k
        self.kernel = kernel
        if self.kernel == 'gaussian':
            self.kernel_ = partial(Gaussian_kernel, sig=sigma)
        elif self.kernel == 'spectrum':
            self.kernel_ = partial(Spectrum_kernel, k=k)
        else:
            raise NotImplementedError(f"Kernel {self.kernel} is not implemented yet")

    def fit(self, X, y):
        """
        inputs:
        - X (size: N_trxd): the points of the training set
        - y (size: N_trx1): the values of the classes
        """
        # We keep values of training in memory for prediction
        N_tr = X.shape[0]
        self.X_tr_ = np.copy(X)

        if self.kernel == 'gaussian':
            K = self.kernel_(X, X, sig=self.sigma)
        elif self.kernel == 'spectrum':
            K = self.kernel_(X, X, k=self.k[0])
            for i in range(len(self.k)-1):
                K+=self.kernel_(X, X, k=self.k[i+1])
        # Define QP and solve it with cvxpy
        alpha = cp.Variable(N_tr)
        objective = cp.Maximize(2*alpha.T@y - cp.quad_form(alpha, K))
        constraints = [0 <= cp.multiply(y,alpha), cp.multiply(y,alpha) <= 1/(2*self.lamb*N_tr)]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        self.alpha_ = alpha.value

        return self

    def predict(self, X):
        """
        inputs:
        - X (size N_texd): the points in R^d we want to classify
        output:
         - the predicted class for the associated y given the
        Linear Regression parameters
        """
        if self.kernel == 'gaussian':
            K_tr_te = self.kernel_(self.X_tr_, X, sig=self.sigma)
        elif self.kernel == 'spectrum':
            K_tr_te = self.kernel_(self.X_tr_, X, k=self.k[0])
            for i in range(len(self.k)-1):
                K_tr_te+=self.kernel_(self.X_tr_, X, k=self.k[i+1])


        return 2 * (self.alpha_.T@K_tr_te > 0).reshape(-1, ).astype("int") - 1

    def score(self, X, y):
        """
        inputs:
        - X (size N_texd): the points in R^d we want to classify
        - y (size N_tex1): the labels of the points
        """
        y_pred = self.predict(X)

        return np.sum(y_pred == y)/y.shape[0]
