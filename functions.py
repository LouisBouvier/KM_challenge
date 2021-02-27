'''
Functions for linear and kernel models
'''

import numpy as np

### ===== functions for Logistic Regression ===== ###

def sigmoid(z):
    """
    input:
    - z (any size): an array-like element
    ouput:
    - the element-wize application of the sigmoïd function on z
    """
    return 1/(1+np.exp(-z))

def compute_loss(X,y,w,b):
    """
    inputs:
    - X (size: Nxd): the points we want to classify
    - y (size: Nx1): the values of the classes
    - w (size: 1xd): the weights of the affine mapping of x
    - b (size: 1x1): the constant of the affine mapping of x
    output:
    - the opposite of the log-likelihood of the Logistic Regression model computed with respect to
    the points (X,y) and the parameters w,b
    """
    X_tilde = np.hstack([X, np.ones((X.shape[0], 1))])
    w_tilde = np.hstack((w,b))
    return -np.sum(y * np.log(sigmoid(w_tilde@X_tilde.T)) + (1-y) * np.log(1-sigmoid(w_tilde@X_tilde.T)), axis=1)

def compute_grad(X,y,w,b):
    """
    inputs:
    - X (size: Nxd): the points we want to classify
    - y (size: Nx1): the values of the classes
    - w (size: 1xd): the weights of the affine mapping of x
    - b (size: 1x1): the constant of the affine mapping of x
    output:
    - the gradient of the loss of the Logistic Regression model computed
    with respect to (w,b) = w_tilde having the points (X,y)
    """
    X_tilde = np.hstack([X, np.ones((X.shape[0], 1))])
    w_tilde = np.hstack((w,b))
    return -X_tilde.T @ (y - sigmoid(w_tilde@X_tilde.T).reshape(-1,))

def compute_hess(X,y,w,b):
    """
    inputs:
    - X (size: Nxd): the points we want to classify
    - y (size: Nx1): the values of the classes
    - w (size: 1xd): the weights of the affine mapping of x
    - b (size: 1x1): the constant of the affine mapping of x
    output:
    - the hessian of the loss of the Logistic Regression model computed
    with respect to (w,b) = w_tilde having the points (X,y)
    """
    X_tilde = np.hstack([X, np.ones((X.shape[0], 1))])
    w_tilde = np.hstack((w,b))
    temp = (sigmoid(w_tilde @ X_tilde.T) * (sigmoid(w_tilde @ X_tilde.T) - 1)).reshape(-1,)
    return -X_tilde.T @ np.diag(temp) @ X_tilde

def backtracking(X,y,w,b,delta,grad,alpha=0.1,beta=0.7):
    """
    inputs:
    - X (size: Nxd): the points we want to classify
    - y (size: Nx1): the values of the classes
    - w (size: 1xd): the weights of the affine mapping of x
    - b (size: 1x1): the constant of the affine mapping of x
    - delta (size n): direction of the search
    - grad (size n): value of the gradient at point (w,b)
    - alpha: factor of the slope of the line in the backtracking line search
    - beta: factor of reduction of the step length

    outputs:
    - t: the step length for the Newton step on the objective function
    computed with backtracking line search towards delta"""

    t = 1
    while(compute_loss(X, y, w+t*delta[:-1], b+t*delta[-1])>
            compute_loss(X,y,w,b) + alpha*t*grad.T @ delta):
        t = beta*t
    return t

def Newton(X, y, w0, b0, eps=pow(10,-1)):
    """
    inputs:
    - X (size: Nxd): the points we want to classify
    - y (size: Nx1): the values of the classes
    - w0 (size: 1xd): the initial weights of the affine mapping of x
    - b0 (size: 1x1): the initial constant of the affine mapping of x
    output:
    - the paramer vector w_tilde_hat = (w_hat, b_hat) which maximizes the log-likelihood of
    the sample (X,y) in the Logistic Regression model (or minimizes the loss)
    - the cached values of the loss evaluated along training
    """
    w_, b_ = w0, b0
    grad = compute_grad(X, y, w0, b0)
    hess = compute_hess(X, y, w0, b0)

#     inv_hess = np.linalg.inv(compute_hess(X,y,w0,b0))
    inv_hess, _, _, _ = np.linalg.lstsq(hess, np.eye(hess.shape[0]))
    dec_2 = grad.T@inv_hess@grad
    Loss_hist = [compute_loss(X,y,w0,b0)]
    while dec_2/2 > eps: # condition on the Newton decrement
        grad = compute_grad(X,y,w_,b_)
        hess = compute_hess(X,y,w_,b_)

#         inv_hess = np.linalg.inv(compute_hess(X,y,w_,b_))
        inv_hess, _, _, _ = np.linalg.lstsq(hess, np.eye(hess.shape[0]))
        dec_2 = grad.T@inv_hess@grad
        delta = - inv_hess@grad
        t_bt = backtracking(X, y, w_, b_, delta, grad)
        w_ = w_ + t_bt*delta[:-1]
        b_ = b_ + t_bt*delta[-1]
        Loss_hist.append(compute_loss(X,y,w_,b_))
    return w_, b_, Loss_hist

def predict_LogReg(x,w,b):
    """
    inputs:
    - x (size 1xd): a point in R^d
    - w (size: 1xd): the weights of the affine mapping of x
    - b (size: 1x1): the constant of the affine mapping of x
    output:
     - the predicted class for the associated y given the
    Logistic Regression parameters
    """
    return (w.T@x + b > 0).astype("int")

### ===== functions for Ridge Regression ===== ###

def compute_RR_MLE(X,y,lamb):
    """
    inputs:
    - X (size: Nxd): the points we want to classify
    - y (size: Nx1): the values of the classes
    outputs:
    - the value of MLE estimation (w_hat, b_hat) in the Linear regression model
    """
    X_tilde = np.vstack((X,np.ones(X.shape[1])))
    temp = np.linalg.inv(X_tilde@X_tilde.T + lamb*X.shape[1]*np.eye(1+X.shape[0]))@X_tilde@y.T
    return temp[:-1], temp[-1]

def predict_RR(x,w,b):
    """
    inputs:
    - x (size 1xd): a point in R^d
    - w (size: 1xd): the weights of the affine mapping of x
    - b (size: 1x1): the constant of the affine mapping of x
    output:
     - the predicted class for the associated y given the
    Linear Regression parameters
    """
    return (w.T@x+b>1/2).astype("int")

### ===== functions for Kernel Ridge Regression ===== ###

def compute_KRR_MLE(X, y, lamb, sig=10):
    """
    inputs:
    - X (size: N_trxd): the points of the training set
    - y (size: N_trx1): the values of the classes
    outputs:
    - the value of MLE estimation (w_hat, b_hat) in the kernel ridge regression model
    """
    K = Gaussian_kernel(X, X, sig=sig)
    alpha = np.linalg.inv(K+lamb*X.shape[1]*np.eye(X.shape[1]))@y.T
    return alpha

def predict_KRR(X_tr, X_te, alpha, sig=10):
    """
    inputs:
    - X_tr (size N_trxd): the points of the training set
    - X_te (size N_texd): the points of the test set we want to classify
    - w (size: 1xd): the weights of the affine mapping
    - b (size: 1x1): the constant of the affine mapping
    output:
     - the predicted class for the associated y_te given the
    Linear Regression parameters
    """
    K_te_tr = Gaussian_kernel(X_tr, X_te, sig=sig)
    return 2*(alpha.T@K_te_tr>0).astype("int")-1
