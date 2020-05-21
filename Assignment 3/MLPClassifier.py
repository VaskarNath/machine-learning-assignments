'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from statistics import mode, mean
from sklearn.model_selection import KFold
from scipy import stats as s
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from sklearn.exceptions import ConvergenceWarning

class MLP(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def mlpclassifier(self, solver, activation, learning_rate, layer_size):
        clf = MLPClassifier(solver=solver, activation=activation, learning_rate_init=learning_rate, hidden_layer_sizes=layer_size)
        clf.fit(self.train_data, self.train_labels)
        return clf


def cross_validation(train_data, train_labels):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    best_solver = ""
    best_activation = ""
    best_learning_rate = 0
    best_layer_size = 0
    best_accuracy = 0
    solvers = ["sgd"]
    activations = ["logistic", "tanh"]
    alphas = [0.01]
    hidden_layer = [(100,)]
    for solver in solvers:
        for activation in activations:
            for alpha in alphas:
                for hidden_layer_size in hidden_layer:
                    accuracy = 0
                    kf = KFold(10, shuffle=True, random_state=89)
                    for train_index, test_index in kf.split(
                            range(len(train_data))):
                        x_train, x_val, y_train, y_val = train_data[train_index], train_data[test_index], train_labels[train_index], train_labels[test_index]
                        mlp = MLP(x_train, y_train).mlpclassifier(solver, activation, alpha, hidden_layer_size)
                        accuracy += mlp.score(x_val, y_val)
                    score = accuracy/10
                    print("Solver: ", solver, " Activation: ", activation, " Learning Rate: ", alpha, " Layer Size: ", hidden_layer_size, " has cross val accuracy: ", score)
                    if score > best_accuracy:
                        best_solver = solver
                        best_activation = activation
                        best_learning_rate = alpha
                        best_layer_size = hidden_layer_size
    return best_solver, best_activation, best_learning_rate, best_layer_size


def classification_accuracy(clf, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    return clf.score(eval_data, eval_labels)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    mlp = MLP(train_data, train_labels)
    solver, activation, learning_rate, layer_size = cross_validation(train_data, train_labels)
    clf = mlp.mlpclassifier(solver, activation, learning_rate, layer_size)
    print("Solver: ", solver, " Activation: ", activation, " Learning Rate: ",
           learning_rate, " Layer Size: ", layer_size)
    print(clf.predict(test_data))
    print("Test accuracy: ",
          classification_accuracy(clf, test_data, test_labels))
    print("Train accuracy: ",
          classification_accuracy(clf, train_data, train_labels))


if __name__ == '__main__':
    main()
