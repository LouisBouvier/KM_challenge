# Machine Learning with Kernel Methods 2021
KKML Kaggle Challenge: https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021/overview 

Team Name: Ker et Paix

Louis Bouvier, Roman Castagné & Julia Linhart

MVA 2020/21 - ENS Paris-Saclay

## Overview:

This repository contains the Python implementation of our solution to the above stated Kaggle Challenge. Our methods make use of several kernels, either working on bags of words or on raw string data. They are briefly described in the provided report, which also indicates a justified ranking.

We have implemented from scratch the following models:
  - Logistic Regression
  - Ridge Regression
  - Kernel Ridge Regression
  - Kernel Support Vector Machines
  - Multiple Kernel Learning
 
 We have dealt with the following kernels:
  - Gaussian kernel on bags of words
  - Spectrum kernel on raw sequences
  - Mismatch kernel on raw sequences
  - Substring kernel on raw sequences
  - Fisher kernel on raw sequences
  - TF-IDF extraction
  - Sum of kernels
  
## Content of the Repository:
  
Models and kernels are implemented from scratch in:
  - ```linear_models.py``` containing the classes ```LogisticRegressor``` and ```RidgeRegressor```
  - ```kernel_models.py``` containg the classes ```KernelRidgeRegressor```, ```KernelSVM``` and ```KernelMKL```
  - ```kernels.py``` containing all the functions nessecary to build the above stated kernels 

Supplementary files:
- ```functions.py```: method-functions for these models (e.g. Newton method used for Logistic Regression) 
- ```utils.py```: utility functions to load precomputed kernels, initialize a model, run the model (training and evalution on all 3 datasets), save the results in a .csv file of right format to be use as submission file

Finally, the Python Notebook ```KM_challenge.ipynb``` contains the code for example runs on our different methods. 


  
  
