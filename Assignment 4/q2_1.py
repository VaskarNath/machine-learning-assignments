'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''

    means = np.zeros((10, 64))
    for i in range(10):
        vector_sum = np.zeros((1, 64))
        num = 0
        for data_index in range(len(train_data)):
            if train_labels[data_index] == i:
                vector_sum += train_data[data_index]
                num += 1
        means[i][:] = vector_sum/num
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        matrix_sum = np.zeros((1, 64, 64))
        num = 0
        for data_index in range(len(train_data)):
            if train_labels[data_index] == i:
                x = train_data[data_index]
                u = means[i][:]
                sub = x - u
                vector = sub.reshape((64, 1))
                matrix_sum += np.dot(vector, np.transpose(vector))
                num += 1
        covariances[i][:][:] = (matrix_sum/num) + 0.01 * np.eye(covariances[i][:][:].shape[0])

    # Compute covariances
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    log_covariances = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        log_covariances.append(np.log(cov_diag).reshape((8, 8)))

    all_concat = np.concatenate(log_covariances, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.savefig("Mean of Log Diagonal Covariance.pdf")
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    det = []
    inv = []
    for i in range(10):
        det.append(np.linalg.det(covariances[i][:][:]))
        inv.append(np.linalg.inv(covariances[i][:][:]))

    result = np.zeros((len(digits), 10))
    for data_index in range(len(digits)):
        row = []
        x = digits[data_index].reshape((1, 64))
        for i in range(10):
            u = means[i][:]
            sub = x - u
            row.append(np.log((2*np.pi)**(-32) * det[i]**(-0.5)) + (-0.5 * np.dot(np.dot(sub, inv[i]), np.transpose(sub))))
        result[data_index][:] = row
    return result


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    likelihood = generative_likelihood(digits, means, covariances)

    result = np.zeros((len(digits), 10))
    for data_index in range(len(digits)):
        row = []
        for i in range(10):
            row.append(likelihood[data_index][i] + np.log(1/10) - np.log((1/10) * sum(np.exp(likelihood[data_index][:]))))
        result[data_index][:] = row
    return result


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''

    cond_likelihood = conditional_likelihood(digits, means, covariances)
    result = 0
    N = len(digits)
    for data_index in range(N):
        result += cond_likelihood[data_index][int(labels[data_index])]
    return result/N


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    result = []
    for data_index in range(len(digits)):
        result.append(np.argmax(cond_likelihood[data_index]))
    return result

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    plot_cov_diagonal(covariances)
    print("Average Condition log-likelihood for Training Set: ",
          avg_conditional_likelihood(train_data, train_labels, means, covariances))
    print("Average Condition log-likelihood for Testing Set: ",
          avg_conditional_likelihood(test_data, test_labels, means, covariances))
    print("Test Accuracy: ", accuracy_score(test_labels, classify_data(test_data, means, covariances)))
    print("Train Accuracy: ", accuracy_score(train_labels, classify_data(train_data, means, covariances)))

if __name__ == '__main__':
    main()
