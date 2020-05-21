'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys
np.set_printoptions(threshold=sys.maxsize)

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    arr = np.concatenate((train_data, train_labels[:, np.newaxis]), axis = 1)
    eta = np.zeros((10, 64))

    for i in range(10):
        selected_rows = arr[arr[:, -1] == i]
        selected_rows = selected_rows[:,:-1]
        eta[i,:] = (np.sum(selected_rows, axis = 0) + 1) / (selected_rows.shape[0] + 2)

    return eta

def plot_images(class_images, name = "q2.2.3.pdf"):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    # Plot the log-diagonal of each covariance matrix side by side
    images = []
    for i in range(10):
        images.append(class_images[i,:].reshape((8,8)))

    all_concat = np.concatenate(images, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.savefig(name, bbox_inches='tight')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for k in range(10):
        generated_data[k,:] = np.random.binomial(1, eta[k,:])

    plot_images(generated_data, "q2.2.4.pdf")

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    def get_prob(row):
        '''
        Compute the generative log-likelihood:
            log p(x|y,mu,Sigma)
        for each class and return an array
        '''
        val = np.sum(row * np.log(eta) + (1-row) * np.log(1 - eta), axis = 1)
        return val

    return np.apply_along_axis(get_prob, axis = 1, arr = bin_digits)

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    log_class_likelihood = generative_likelihood(bin_digits, eta)
    log_prior = np.log(0.1) * np.ones(log_class_likelihood.shape)
    log_evidence = np.mean(np.exp(log_class_likelihood), axis = 1) # Just a vector
    log_evidence = np.log(np.tile(log_evidence, (10, 1)).T) # Create matrix with columns being log_evidence

    return log_class_likelihood + log_prior - log_evidence

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    total = 0
    N = len(bin_digits)
    for i in range(N):
        total += cond_likelihood[i, int(labels[i])]
    return total / N

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    classifications = np.argmax(cond_likelihood, axis = 1)
    return classifications

def conditional_likelihood2(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    likelihood = generative_likelihood(bin_digits, eta)

    result = np.zeros((len(bin_digits), 10))
    for data_index in range(len(bin_digits)):
        row = []
        for i in range(10):
            row.append(likelihood[data_index][i] + np.log(1 / 10) - np.log((1/10) *
                sum(np.exp(likelihood[data_index][:]))))
        result[data_index][:] = row
    return result

def generative_likelihood2(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''

    result = np.zeros((len(bin_digits), 10))
    for data_index in range(len(bin_digits)):
        row = []
        b = bin_digits[data_index]
        for k in range(10):
            row.append(np.dot(b, np.log(eta[k][:])) + np.dot(1 - b, np.log(1 - eta[k][:])))
        result[data_index] = row
    return result

def avg_conditional_likelihood2(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    result = 0
    N = len(bin_digits)
    for data_index in range(N):
        result += cond_likelihood[data_index][int(labels[data_index])]
    return result / N

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # 2.2.3
    plot_images(eta)

    # 2.2.4
    generate_new_data(eta)

    # 2.2.5
    print("The average conditional log-likelihood for the training set is ", avg_conditional_likelihood(train_data, train_labels, eta))
    print("The average conditional log-likelihood for the testing set is ", avg_conditional_likelihood(test_data, test_labels, eta))

    # 2.2.6
    print("The test accuracy is ", accuracy_score(test_labels, classify_data(test_data, eta)))
    print("The train accuracy is ", accuracy_score(train_labels, classify_data(train_data, eta)))

if __name__ == '__main__':
    main()
