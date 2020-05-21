'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, accuracy_score, precision_score, mean_squared_error, confusion_matrix, recall_score


def MLP(train_data, train_labels):
    '''
    Train a SVM by tuning the right hyper-parameter to maximize accuracy of the
    estimator
    '''
    parameters = [{'activation': ['logistic'], 'solver': ["sgd"],
                         'learning_rate_init': [0.01, 0.001, 0.0001], "hidden_layer_sizes": [(100,)]},
                  {'activation': ["tanh"], 'solver': ["sgd"],
                   'learning_rate_init': [0.01, 0.001, 0.0001], "hidden_layer_sizes": [(100,)]}]
    clf = GridSearchCV(MLPClassifier(), parameters)
    clf.fit(train_data, train_labels)
    means = clf.cv_results_['mean_test_score']
    for mean, params in zip(means, clf.cv_results_['params']):
        print("Accuracy: %0.3f Parameters: %r" % (mean, params))
    return clf


def classification_accuracy(clf, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    return clf.score(eval_data, eval_labels)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    clf = MLP(train_data, train_labels)
    predicted_labels = clf.predict(train_data)
    fpr_rf, tpr_rf, _ = roc_curve(test_data, predicted_labels)

    print("Best Parameters: ", clf.best_params_)
    print("Test accuracy: ",
          classification_accuracy(clf, test_data, test_labels))
    print("Train accuracy: ",
          classification_accuracy(clf, train_data, train_labels))


if __name__ == '__main__':
    main()
