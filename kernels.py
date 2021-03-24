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

ALPHABET = ["A", "T", "G", "C"]

### ================ Gaussian Kernel =================== ###
def Gaussian_kernel(X1, X2, sig, **args):
    """inputs:
    - X1 (size N1xd): a set of points
    - X2 (size N2xd): another one
    - sig (float): the std of the kernel
    ouput:
    - the associated (N1xN2) Gaussian kernel
    """
    return np.exp(-distance_matrix(X1,X2)/(2*sig**2))

### ================ Spectrum Kernel =================== ###

def spectrum(x, k):
    l = len(x)
    spectrum_x = np.array([x[i:(i + k)] for i in range(l - k + 1)])
    return np.array(spectrum_x)

def Spectrum_kernel(X1, X2, k, **args):
    """inputs:
    - X1 (size N1xd): a set of sequences
    - X2 (size N2xd): another one
    - k (int): the length of the substrings
    ouput:
    - the associated (N1xN2) Spectrum kernel
    """
    ### ------ OLD COMPUTATION OF SPECTRUM KERNEL ------
    # # substrings: all possible combinations of A,T,G,C of length k
    # A_k = [''.join(s) for s in product(["A", "T", "G", "C"], repeat=k)]
    #
    # # nb of occurances of the elements of A_k in the k-spectrum of X1 (resp. X2)
    # phi_spect_X1 = np.array([[np.sum(spectrum(x,k)==u) for u in A_k] for x in X1])
    # phi_spect_X2 = np.array([[np.sum(spectrum(x,k)==u) for u in A_k] for x in X2])
    ### ------------------ END ---------------------

    A_k = {''.join(s): i for i, s in enumerate(product(["A", "T", "G", "C"], repeat=k))}
    phi_spect_X1 = np.zeros((len(X1), len(A_k)))
    phi_spect_X2 = np.zeros((len(X2), len(A_k)))
    for i, x in enumerate(X1):
        for j in range(len(x) - k + 1):
            phi_spect_X1[i][A_k[x[j:(j + k)]]] += 1
    for i, x in enumerate(X2):
        for j in range(len(x) - k + 1):
            phi_spect_X2[i][A_k[x[j:(j + k)]]] += 1

    K_s = phi_spect_X1 @ phi_spect_X2.T
    K_s = K_s + np.eye(K_s.shape[0], K_s.shape[1]) * pow(10,-12)
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
                if min(i_str_1, i_str_2) >= l-1:

                    if i_str_1 == 0 and i_str_2 == 0:
                        continue

                    if i_str_2 == 0:
                        # Computation of B
                        B_temp[l][i_str_1, i_str_2] = lambd * B_temp[l][i_str_1-1, i_str_2]
                        K_temp[l][i_str_1, i_str_2] = K_temp[l][i_str_1-1, i_str_2]

                    elif i_str_1 == 0:
                        B_temp[l][i_str_1, i_str_2] = lambd * B_temp[l][i_str_1, i_str_2-1]
                        K_temp[l][i_str_1, i_str_2] = K_temp[l][i_str_1, i_str_2-1]

                    else:
                        B_temp[l][i_str_1, i_str_2] = lambd * B_temp[l][i_str_1-1, i_str_2] \
                                                    + lambd * B_temp[l][i_str_1, i_str_2-1] \
                                                    - lambd ** 2 * B_temp[l][i_str_1-1, i_str_2-1] \
                                                    + lambd ** 2 * int(a == b) * B_temp[l-1][i_str_1-1, i_str_2-1]

                        # This corresponds to the sum in the computation of K
                        K_temp[l][i_str_1, i_str_2] = K_temp[l][i_str_1-1, i_str_2] \
                                                    + K_temp[l][i_str_1, i_str_2-1] \
                                                    - K_temp[l][i_str_1-1, i_str_2-1] \
                                                    + lambd ** 2 * int(a == b) * B_temp[l-1][i_str_1-1, i_str_2-1]

    return K_temp[k][-1, -1]

def substring_kernel(X1, X2, k, lambd, **args):
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

### ================ Ficher Kernel for ungapped HMM =================== ###
def spectrum_matrix(X,k):
    X_spect =[]
    for x in X:
        l = len(x)
        spect_x = np.array([x[i:(i + k)] for i in [j for j in range(l-k+1) if j%k==0]])
        X_spect.append(spect_x.reshape(-1,1))
    return np.array(X_spect)

def emission_probs(spectrum_matrix,chars):
    #state = group of k subsequent characters in the sequence
    # for each state we assign an emission probability of each subsequence (nb occurences)
    nb_samples = spectrum_matrix.shape[0]
    nb_states = spectrum_matrix.shape[1]
    nb_chars = len(chars) # number of possible emissions
    probs = np.zeros((nb_states,nb_chars))
    for i in range(nb_states):
        for j in range(nb_chars):
            probs[i][j]=np.sum(spectrum_matrix[:,i] == chars[j])/nb_samples
    return probs

def probs_x(spect,probs,chars):
    n_states = len(spect)
    p = []
    for i in range(n_states):
        idx = np.where(chars == spect[i])[0][0]
        p.append(probs[i][idx])
    return np.array(p)

def log_likelihood(probs_x):
    ll = 0
    for p in probs_x:
        ll+=np.log(p)
    return ll

def fisher_score(probs_x):
    eps = 10e-10
    phi = []
    for p in probs_x:
      phi.append(1/(p+eps))
    return np.array(phi).reshape(-1,1)

def Fisher_kernel(X1,X2,X_HMM,k, **args):
    chars = np.array([''.join(s) for s in product(["A", "T", "G", "C"], repeat=k)])
    spectrum_matrix_HMM = spectrum_matrix(X_HMM,k)
    em_probs = emission_probs(spectrum_matrix_HMM,chars)

    spectrum_matrix_X1 = spectrum_matrix(X1,k)
    spectrum_matrix_X2 = spectrum_matrix(X2,k)

    K = np.zeros((X1.shape[0],X2.shape[0]))
    norm = 0
    for i in range(X1.shape[0]):
        spect_x1 = spectrum_matrix_X1[i]
        probs_x1 = probs_x(spect_x1,em_probs,chars)
        phi_x1 = fisher_score(probs_x1)
        norm += phi_x1.T @ phi_x1/X1.shape[0]
        for j in range(X2.shape[0]):
            spect_x2 = spectrum_matrix_X2[j]
            probs_x2 = probs_x(spect_x2,em_probs,chars)
            phi_x2 = fisher_score(probs_x2)
            K[i][j] = phi_x1.T @ phi_x2
    return K/norm

### ================ Mismatch kernel =================== ###

class Node(object):
    def __init__(self, parent, letter):
        self.parent = parent
        self.letter = letter
        self.sequence = None
        self.pointers = {}

    def get_sequence(self):
        return self.sequence

    def get_parent(self):
        return self.parent

    def get_pointers(self):
        return self.pointers

    def set_sequence(self):
        if self.parent:
            self.sequence = self.parent.get_sequence()+self.letter if self.parent.get_sequence() else self.letter

    def set_pointers(self, dataset, depth, max_mismatch):
        Pointers = {}
        if self.get_parent() is not None:
            parent_Pointers = self.get_parent().get_pointers()
            for pointer, mismatch in parent_Pointers.items():
                if dataset[pointer][depth]!=self.letter:
                    new_mismatch = mismatch+1
                    if new_mismatch <= max_mismatch:
                        Pointers[pointer] = new_mismatch
                else:
                    Pointers[pointer] = mismatch
        else:
            for i in range(len(dataset)):
                Pointers[i] = 0
        self.pointers = Pointers


class Tree(Node):
    def __init__(self,k,m):
        self.maxdepth = k
        self.max_nb_mismatches = m
        # create the strings of length k with the alphabet
        self.A_k = {''.join(s): i for i,s in enumerate(product(ALPHABET, repeat=k))}
        # create the root node with no letter and no parent
        root = Node(parent=None, letter=None)
        root.set_pointers(list(self.A_k.keys()), 0, self.max_nb_mismatches)
        # create a dictionnary of nodes: width first
        self.Nodes = {0: [root]}
        for d in range(1,self.maxdepth+1):
            self.Nodes[d] = []
        for count in range(self.maxdepth):
            for parent_ in self.Nodes[count]:
                for charact in ALPHABET:
                    child = Node(parent_,charact)
                    child.set_sequence()
                    child.set_pointers(list(self.A_k.keys()), count, self.max_nb_mismatches)
                    self.Nodes[count+1].append(child)

    def get_Nodes(self):
      return self.Nodes

    def build_kernel(self, X1, X2):
        sub_strings1 = np.zeros((X1.shape[0], len(X1[0]) - self.maxdepth + 1))
        for i, x in enumerate(X1):
            for j in range(len(x)-self.maxdepth+1):
                sub_strings1[i, j] = self.A_k[x[j:j+self.maxdepth]]

        sub_strings2 = np.zeros((X2.shape[0], len(X2[0]) - self.maxdepth + 1))
        for i, x in enumerate(X2):
            for j in range(len(x)-self.maxdepth+1):
                sub_strings2[i, j] = self.A_k[x[j:j+self.maxdepth]]

        K = np.zeros((X1.shape[0], X2.shape[0]))
        for leaf in self.get_Nodes()[self.maxdepth]:
            leaf_pointed = set(leaf.get_pointers().keys())
            K += get_occurences(leaf_pointed, sub_strings1, sub_strings2)
        return K

@jit(nopython=True)
def get_occurences(leaf_pointed, sub_strings1, sub_strings2):
    occurences_1 = np.zeros((sub_strings1.shape[0], sub_strings1.shape[1]))
    occurences_2 = np.zeros((sub_strings2.shape[0], sub_strings2.shape[1]))
    for i, line in enumerate(sub_strings1):
        for j, el in enumerate(line):
            occurences_1[i,j] = int(el in leaf_pointed)
    for i, line in enumerate(sub_strings2):
        for j, el in enumerate(line):
            occurences_2[i,j] = int(el in leaf_pointed)

    #occurences_1 = np.array([[1 if el in leaf_pointed else 0 for el in sub_strings1[k]] for k in range(len(sub_strings1))])
    #occurences_2 = np.array([[1 if el in leaf_pointed else 0 for el in sub_strings2[k]] for k in range(len(sub_strings2))])
    occurences_1 = np.sum(occurences_1, axis=1)
    occurences_2 = np.sum(occurences_2, axis=1)
    #occurences_1 = np.isin(sub_strings1, leaf_pointed).sum(axis=1)
    #occurences_2 = np.isin(sub_strings2, leaf_pointed).sum(axis=1)
    occurences_1 = occurences_1.reshape(-1,1)
    occurences_2 = occurences_2.reshape(-1,1)
    return occurences_1@occurences_2.T


def Mismatch_kernel(X1, X2, k, m):
    """inputs:
    - X1 (size N1xd): a set of strings
    - X2 (size N2xd): another one
    - k (integer): the length of the substrings considered
    - m (integer): the order of mismatch accepted
    ouput:
    - the associated (N1)x(N2) mismatch kernel
    """
    Test_tree = Tree(k=k, m=m)
    kernel = Test_tree.build_kernel(X1,X2)
    return kernel


## ===================== TF-IDF =====================
def compute_TF(x, k, A_k):
    tf = np.zeros(len(A_k))
    for i in range(len(x) - k + 1):
        tf[A_k[x[i:i+k]]] += 1
    return tf

def compute_IDF(tf):
    res = np.zeros(tf.shape[1])
    term_freq = np.sum(tf > 0, axis=0)
    res[term_freq > 0] = np.log(tf.shape[0] / term_freq[term_freq > 0])
    return res.reshape(1, tf.shape[1])

def compute_TFIDF(X, k=3, idf=None, bool_return_idf=False):
    """
    Computes TFIDF representations of a list of strings. The strings are biological sequences.
    Since we don't have access to words for biological sequences, the words will be patterns of k characters
    (e.g. ATG or TTC if k=3)

    Inputs:
        - X (array of strings): sequences to encode.
        - k (int): size of the "words".
        - idf (array): a precomputed IDF, used for the testing set.
        - bool_return_idf (bool): whether to return the computed IDF array.
    Output:
        - (array): the TF-IDF representations for each of the sequences in X.
        - (array, optional): the IDF.

    """
    A_k = {''.join(s): i for i, s in enumerate(product(["A", "T", "G", "C"], repeat=k))}

    tf = np.zeros((X.shape[0], len(A_k)))
    for i, x in enumerate(X):
        tf[i] = compute_TF(x, k, A_k)

    if idf is None:
        idf = compute_IDF(tf)

    if bool_return_idf:
        return tf * idf, idf
    else:
        return tf * idf

def tfidf_kernel(X1, X2, k=3):

    phi1, idf = compute_TFIDF(X1, k=k, bool_return_idf=True)
    phi2 = compute_TFIDF(X2, k=k, idf=idf)

    return phi1 @ phi2.T
