'''
Kernels:
- Gaussian kernel
- Spectrum kernel
- Mismatch kernel
- Substring kernel
'''

import numpy as np
from itertools import product
from numba import jit
from scipy.spatial import distance_matrix

### ================ Gaussian Kernel =================== ###
def Gaussian_kernel(X1, X2, sig):
    """inputs:
    - X1 (size N1xd): a set of points
    - X2 (size N2xd): another one
    - sig (float): the std of the kernel
    ouput:
    - the associated (N1xN2) Gaussian kernel
    """
    return np.exp(-distance_matrix(X1,X2)/(2*sig**2))

### ================ Spectrum Kernel =================== ###

def spectrum(x,k):
    l = len(x)
    spectrum_x = np.array([x[i:(i + k)] for i in range(l - k + 1)])
    return np.array(spectrum_x)

def Spectrum_kernel(X1, X2, k):
    """inputs:
    - X1 (size N1xd): a set of sequences
    - X2 (size N2xd): another one
    - k (int): the length of the substrings
    ouput:
    - the associated (N1xN2) Spectrum kernel
    """
    # substrings: all possible combinations of A,T,G,C of length k
    A_k = [''.join(s) for s in product(["A", "T", "G", "C"], repeat=k)]

    # nb of occurances of the elements of A_k in the k-spectrum of X1 (resp. X2)
    phi_spect_X1 = np.array([[np.sum(spectrum(x,k)==u) for u in A_k] for x in X1])
    phi_spect_X2 = np.array([[np.sum(spectrum(x,k)==u) for u in A_k] for x in X2])

    K_s = phi_spect_X1 @ phi_spect_X2.T
    K_s = K_s + np.eye(K_s.shape[0], K_s.shape[1])*pow(10,-12)
    return K_s

### ================ Substring Kernel =================== ###

@jit(nopython=True)
def substring_similarity(seq_1, seq_2, k, lambd):
    # Initialize K and B
    K_temp = [np.ones((len(seq_1), len(seq_2)))]
    K_temp += [np.zeros((len(seq_1), len(seq_2))) for _ in range(k)]

    B_temp = [np.ones((len(seq_1), len(seq_2)))]
    B_temp += [np.zeros((len(seq_1), len(seq_2))) for _ in range(k)]

    for l in range(1, k+1):
        # First, loop over the first sequence
        for i_str_1 in range(len(seq_1)):

            # Then, loop over the second sequence
            for i_str_2 in range(len(seq_2)):
                a = seq_1[i_str_1]
                b = seq_2[i_str_2]

                # If min < l, then the matrix already has zeros in the right place : we can continue
                if min(i_str_1, i_str_2) >= l:

                    # Computation of B
                    B_temp[l][i_str_1, i_str_2] = lambd * B_temp[l][i_str_1-1, i_str_2] \
                                                + lambd * B_temp[l][i_str_1, i_str_2-1] \
                                                - lambd ** 2 * B_temp[l][i_str_1-1, i_str_2-1] \
                                                + lambd ** 2 * int(a == b) * B_temp[l-1][i_str_1-1, i_str_2-1]

                    # This corresponds to the sum in the computation of K
                    # !!!!! This one is probably wrong !!!!!
#                     K_sum_1 = 0
#                     for j_prime in range(i_str_2+1):
#                         if seq_2[j_prime] == a:
#                             K_sum_1 += B_temp[l-1][i_str_1-1, j_prime-1]

#                     # Computation of K
#                     K_temp[l][i_str_1, i_str_2] = K_temp[l][i_str_1-1, i_str_2] + lambd ** 2 * K_sum_1

                    # This one is wrong too, but less
                    K_temp[l][i_str_1, i_str_2] = K_temp[l][i_str_1-1, i_str_2] \
                                                + K_temp[l][i_str_1, i_str_2-1] \
                                                - K_temp[l][i_str_1-1, i_str_2-1] \
                                                + lambd ** 2 * int(a == b) * B_temp[l-1][i_str_1-1, i_str_2-1]

    return K_temp[k][-1, -1]

def substring_kernel(X1, X2, k, lambd):
    """

    """
    assert all([type(x) == str for x in X1]), "not a list of strings"
    assert all([type(x) == str for x in X2]), "not a list of strings"

    K = - np.ones((len(X1), len(X2)))

    for i, seq_1 in enumerate(X1):
        for j, seq_2 in enumerate(X2):
            # This basically corresponds to filling the lower diagonal
            if seq_2 in X1 and seq_1 in X2 and K[np.where(X1 == seq_2), np.where(X2 == seq_1)] != -1:
                K[i, j] = K[np.where(X1 == seq_2), np.where(X2 == seq_1)]
            else:
                K[i, j] = substring_similarity(seq_1, seq_2, k, lambd)

    return K