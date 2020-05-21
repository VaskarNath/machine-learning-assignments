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
    arr = np.concatenate((train_data, train_labels[:, np.newaxis]), axis = 1)
    means = np.zeros((10, 64))

    # Compute means
    for i in range(10):
        selected_rows = arr[arr[:, -1] == i]
        selected_rows = selected_rows[:,:-1]
        means[i,:] = np.mean(selected_rows, axis = 0)

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    arr = np.concatenate((train_data, train_labels[:, np.newaxis]), axis = 1)
    means = compute_mean_mles(train_data, train_labels)
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for i in range(10):
        selected_rows = arr[arr[:, -1] == i]
        selected_rows = selected_rows[:,:-1]
        mu = (means[i,:])[:, np.newaxis]
        cov = np.zeros((64,64))
        for j in range(selected_rows.shape[0]):
            row = (selected_rows[j,:])[:,np.newaxis]
            cov += np.dot((row - mu), (row - mu).T)
        cov /= selected_rows.shape[0]
        covariances[i,:,:] = cov + 0.01 * np.eye(cov.shape[0])

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    log_diagonals = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        log_diagonals.append(np.log(cov_diag).reshape((8, 8)))

    all_concat = np.concatenate(log_diagonals, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.savefig("q2.1.1.pdf", bbox_inches='tight')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    constant = -0.5 * digits.shape[1] * np.log(2*np.pi)
    cov_det_logs = [-0.5 * np.log(np.linalg.det(covariances[k,:,:])) for k in range(10)]
    cov_inv = [np.linalg.inv(covariances[k,:,:]) for k in range(10)]

    def get_prob_k(row, k):
        '''
        Compute the generative log-likelihood:
            log p(x|y,mu,Sigma)
        for class k.
        '''
        arr = row.reshape((row.shape[0], 1))
        det_log = cov_det_logs[k]
        inv = cov_inv[k]
        mean = means[k,:][:,np.newaxis]
        x = arr - mean
        val = constant + det_log - 0.5 * np.dot(np.dot(x.T, inv), x)
        return val.item()

    def get_prob(row):
        '''
        Compute the generative log-likelihood:
            log p(x|y,mu,Sigma)
        for each class and return an array
        '''
        probs = [get_prob_k(row,k) for k in range(10)]
        return np.array(probs)

    return np.apply_along_axis(get_prob, axis = 1, arr = digits)

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    log_class_likelihood = generative_likelihood(digits, means, covariances)
    log_prior = np.log(0.1) * np.ones(log_class_likelihood.shape)
    log_evidence = np.mean(np.exp(log_class_likelihood), axis = 1) # Just a vector
    log_evidence = np.log(np.tile(log_evidence, (10, 1)).T) # Create matrix with columns being log_evidence

    return log_class_likelihood + log_prior - log_evidence

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    total = 0
    N = len(digits)
    for i in range(N):
        total += cond_likelihood[i, int(labels[i])]
    return total / N

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    classifications = np.argmax(cond_likelihood, axis = 1)
    return classifications



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # 2.1.1
    plot_cov_diagonal(covariances)

    # 2.1.2
    print("The average conditional log-likelihood for the training set is ", avg_conditional_likelihood(train_data, train_labels, means, covariances))
    print("The average conditional log-likelihood for the testing set is ", avg_conditional_likelihood(test_data, test_labels, means, covariances))

    # 2.1.3
    print("The test accuracy is ", accuracy_score(test_labels, classify_data(test_data, means, covariances)))
    print("The train accuracy is ", accuracy_score(train_labels, classify_data(train_data, means, covariances)))



if __name__ == '__main__':
    main()
