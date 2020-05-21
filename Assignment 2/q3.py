from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from math import exp
from scipy.special import logsumexp
from scipy.special import softmax
from sklearn.model_selection import KFold
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    N_train = len(x_train)
    A = np.zeros(shape=(N_train, N_train))
    dist = l2(x_train, np.transpose(test_datum))
    dist = dist * (-1/(2 * (tau ** 2)))
    dist = softmax(dist)
    for i in range(N_train):
        A[i][i] = dist[i]
    A = A + 1e-8 * np.eye(A.shape[0])
    w = np.linalg.solve(np.dot(np.dot(np.transpose(x_train), A), x_train),
                        np.dot(np.dot(np.transpose(x_train), A), y_train))
    return np.dot(np.transpose(test_datum), w)
    ## TODO

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = 0.5 * ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses

#to implement
def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    result = np.zeros(shape=(len(taus), k))
    kf = KFold(k, shuffle=True, random_state=23)
    i = 0
    for train_index, test_index in kf.split(range(len(x))):

        x_train, x_test, y_train, y_test = x[train_index], x[test_index],\
        y[train_index], y[test_index]
        result[:, i] = run_on_fold(x_test, y_test, x_train, y_train, taus)
        i = i + 1
    return result
    ## TODO


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 100)
    losses = run_k_fold(x, y, taus, k=5) # Wasnt sure whether to return the averaged value of different cv runs or the entire (200, 5) matrix
    average_arr = []                     # So I returned the entire matrix and averaged the rows after
    for i in range(len(losses)):
        average_arr.append(losses[i].mean())
    plt.plot(taus, average_arr)
    plt.xlabel("Tau")
    plt.ylabel("Average Loss under Cross Validation")
    plt.title("Average Loss vs. Tau Value")
    plt.show()
    print("min loss = {}".format(losses.min()))

