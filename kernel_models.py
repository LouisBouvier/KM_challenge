'''
Kernel Models:
- kernel Ridge Regression
- kernel SVM
'''

import numpy as np
import cvxpy as cp
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin

from kernels import Gaussian_kernel, Spectrum_kernel, substring_kernel, Fisher_kernel


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

    def __init__(self, lamb=1., sigma=1., k = 3, X_HMM= None, kernel=None, precomputed_kernel=None):
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
        self.params = {'lamb': lamb, 'sig': sigma, 'k': k}
        self.X_HMM = X_HMM
        if precomputed_kernel is not None:
            self.kernel_ = precomputed_kernel
        elif self.kernel == 'gaussian':
            self.kernel_ = partial(Gaussian_kernel, sig=sigma)
        elif self.kernel == 'spectrum':
            self.kernel_ = partial(Spectrum_kernel, k=k)
        elif self.kernel == 'substring':
            warnings.warn("Computing the subtring kernel on the fly is computationnally heavy, you should probably precompute it.")
            self.kernel_ = partial(substring_kernel, k=k)
        elif self.kernel == 'fisher':
            self.kernel_ = partial(Fisher_kernel, k=k)
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

        # if self.kernel == 'gaussian':
        #     K = self.kernel_(X, X, sig=self.sigma)
        # elif self.kernel == 'spectrum':
        #     K = self.kernel_(X, X, k=self.k[0])
        #     for i in range(len(self.k)-1):
        #         K+=self.kernel_(X, X, k=self.k[i+1])
        if self.kernel =='fisher':
            K = self.kernel_(X, X, self.X_HMM, **self.params)
            K+= 1e-8
        else:
            K = self.kernel_(X, X, **self.params)

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
        # if self.kernel == 'gaussian':
        #     K_tr_te = self.kernel_(self.X_tr_, X, sig=self.sigma)
        # elif self.kernel == 'spectrum':
        #     K_tr_te = self.kernel_(self.X_tr_, X, k=self.k[0])
        #     for i in range(len(self.k)-1):
        #         K_tr_te+=self.kernel_(self.X_tr_, X, k=self.k[i+1])
        if self.kernel == 'fisher':
            K_tr_te = self.kernel_(self.X_tr_, X, self.X_HMM, **self.params)
            K_tr_te+= 1e-8
        else:
            K_tr_te = self.kernel_(self.X_tr_, X, **self.params)

        return 2 * (self.alpha_.T@K_tr_te > 0).reshape(-1, ).astype("int") - 1

    def score(self, X, y):
        """
        inputs:
        - X (size N_texd): the points in R^d we want to classify
        - y (size N_tex1): the labels of the points
        """
        y_pred = self.predict(X)

        return np.sum(y_pred == y)/y.shape[0]


def simplex_projection(eta):
    # See https://lcondat.github.io/publis/Condat_simplexproj.pdf, Algorithm 1 for an explanation of this function
    u = np.sort(eta)[::-1]
    tmp = (np.cumsum(u) - 1) / (np.arange(len(eta)) + 1)
    nonzero = np.nonzero(tmp < u)[0]
    if len(nonzero) > 0:
        K = nonzero[-1]
    else:
        K = -1
    tau = tmp[K]
    return np.maximum(eta - tau, 0)


class KernelMKL(object):

    def __init__(self, lamb, kernels, get_precomputed_kernels, step, n_iterations=1):
        """
        inputs:
            - lamb: lambda parameter for the SVM
            - kernels: dict of list of kernels to use for training and testing. This datastructure should have the following format:
                        - "train"
                        | -- (array) kernel 1
                        | -- (array) kernel 2
                        | -- ...
                        - "eval"
                        | -- (array) kernel 1
                        | -- (array) kernel 2
                        | -- ...
            - get_precomputed_kernels: function that takes in arguments `K_tr` and `K_ev` and returns a precomputed_kernel.
            - step: gradient descent step
            - n_iterations: number of iterations for the projected gradient algorithm.
        """
        self.lamb = lamb
        self.kernels = kernels
        self.get_precomputed_kernels = get_precomputed_kernels
        self.step = step
        self.n_iterations = n_iterations

        assert len(kernels["train"]) == len(kernels["eval"])
        self.n_kernels = len(kernels["train"])
        self.eta = np.ones(self.n_kernels) / self.n_kernels

    def fit(self, X, y, tr_idx):

        for _ in range(self.n_iterations):
            # compute weighted sum of kernels
            K_tr = sum([self.eta[i] * self.kernels["train"][i] for i in range(self.n_kernels)])

            precomputed_kernel = self.get_precomputed_kernels(K_tr=K_tr)

            # compute your objective function by fitting a SVM
            model = KernelSVM(lamb=self.lamb, precomputed_kernel=precomputed_kernel)
            model.fit(X, y)

            # gradient descent step
            grad = np.zeros(self.n_kernels)
            for i in range(self.n_kernels):
                grad[i] = -1/2 * model.alpha_.T@self.kernels["train"][i][np.ix_(tr_idx, tr_idx)]@model.alpha_
            self.eta -= self.step * grad

            # projection of the new eta to the simplex
            self.eta = simplex_projection(self.eta)

        # fit your model with your final parameters. We also load the evaluation kernel so that we can run directly functions from the SVM class
        K_tr = sum([self.eta[i] * self.kernels["train"][i] for i in range(self.n_kernels)])
        K_ev = sum([self.eta[i] * self.kernels["eval"][i] for i in range(self.n_kernels)])
        precomputed_kernel = self.get_precomputed_kernels(K_tr=K_tr, K_ev=K_ev)
        self.model = KernelSVM(lamb=self.lamb, precomputed_kernel=precomputed_kernel)
        self.model.fit(X, y)

    def predict(self, X):
        """
        inputs:
        - X (size N_texd): the points in R^d we want to classify
        output:
         - the predicted class for the associated y given the
        Linear Regression parameters
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        inputs:
        - X (size N_texd): the points in R^d we want to classify
        - y (size N_tex1): the labels of the points
        """
        return self.model.score(X, y)
