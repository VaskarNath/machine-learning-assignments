'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

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
    eta = np.zeros((10, 64))
    b = binarize_data(train_data)
    for data_index in range(len(train_data)):
        k = int(train_labels[data_index])
        eta[k][:] += b[data_index]
    return (eta + 1) / 702


def plot_images(class_images, title):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''

    temp = []
    for i in range(10):
        temp.append(class_images[i].reshape((8, 8)))


    all_concat = np.concatenate(temp, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.savefig(title)
    plt.show()
    plt.close()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))

    for k in range(10):
        row = []
        for j in range(64):
            b_j = np.random.binomial(1, eta[k][j])
            row.append(b_j)
        generated_data[k] = row
    plot_images(binarize_data(generated_data), "generated data.pdf")


def generative_likelihood(bin_digits, eta):
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

def conditional_likelihood(bin_digits, eta):
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

def avg_conditional_likelihood(bin_digits, labels, eta):
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


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    # Compute and return the most likely class
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    result = []
    for data_index in range(len(bin_digits)):
        result.append(np.argmax(cond_likelihood[data_index]))
    return result


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta, "eta.pdf")
    generate_new_data(eta)

    print("Average Condition log-likelihood for Training Set: ",
          avg_conditional_likelihood(train_data, train_labels, eta))
    print("Average Condition log-likelihood for Testing Set: ",
          avg_conditional_likelihood(test_data, test_labels, eta))
    print("Test Accuracy: ", accuracy_score(test_labels, classify_data(test_data, eta)))
    print("Train Accuracy: ", accuracy_score(train_labels, classify_data(train_data, eta)))


if __name__ == '__main__':
    main()
