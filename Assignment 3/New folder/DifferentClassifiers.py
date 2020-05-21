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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, mean_squared_error, confusion_matrix, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_confusion_matrix


def AdaBoost(train_data, train_labels):
    '''
    Train a SVM by tuning the right hyper-parameter to maximize accuracy of the
    estimator
    '''
    parameters = [{"base_estimator": [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=8)],
                   'n_estimators': [50, 75, 100, 150],
                         'learning_rate': [0.5, 0.4, 0.75, 1]}]
    clf = GridSearchCV(AdaBoostClassifier(), parameters)
    clf.fit(train_data, train_labels)
    means = clf.cv_results_['mean_test_score']
    for mean, params in zip(means, clf.cv_results_['params']):
        print("Accuracy: %0.3f Parameters: %r" % (mean, params))
    return clf


def MLP(train_data, train_labels):
    '''
    Train a SVM by tuning the right hyper-parameter to maximize accuracy of the
    estimator
    '''
    parameters = [{'activation': ['logistic', "tanh"], 'solver': ["adam"],
                         'learning_rate_init': [0.01, 0.001], "hidden_layer_sizes": [(50,), (50, 50)]}]
    clf = GridSearchCV(MLPClassifier(), parameters)
    clf.fit(train_data, train_labels)
    means = clf.cv_results_['mean_test_score']
    for mean, params in zip(means, clf.cv_results_['params']):
        print("Accuracy: %0.3f Parameters: %r" % (mean, params))
    return clf


def SVM(train_data, train_labels):
    '''
    Train a SVM by tuning the right hyper-parameter to maximize accuracy of the
    estimator
    '''
    parameters = [{'kernel': ['rbf', 'sigmoid', 'linear'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000], 'probability': [True]}]
    clf = GridSearchCV(SVC(), parameters)
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


def plot_roc_curve(clf, test_data, test_labels, type):
    y_score = clf.predict_proba(test_data)
    y_test = label_binarize(test_labels, classes=range(10))

    # Compute ROC curve and ROC area for each class
    false_positive_rate = {}
    true_positive_rate = {}
    roc_auc = {}
    for i in range(10):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])
    for i in range(10):
        plt.plot(false_positive_rate[i], true_positive_rate[i], label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(type + ': ROC Curve for each of the 10 Classes of Handwritten Digits')
    plt.legend(loc="lower right")
    plt.savefig(type + " ROC Curve.pdf")
    plt.close()


def print_accuracy_metric(clf, test_data, test_labels, train_data, train_labels):
    print("Best Parameters: ", clf.best_params_)
    print("Test accuracy: ",
          classification_accuracy(clf, test_data, test_labels))
    print("Train accuracy: ",
          classification_accuracy(clf, train_data, train_labels))


def make_confusion_matrix(clf, x_test, y_test, type):
    disp = plot_confusion_matrix(clf, x_test, y_test,
                                 display_labels=range(10),
                                 cmap=plt.cm.Blues, values_format='.3g')
    disp.ax_.set_title("Confusion Matrix")
    print(disp.confusion_matrix)
    plt.savefig(type + " Confusion Matrix.pdf")
    plt.close()


def get_precision_score(y_pred, y_true):
    for digit, score in enumerate(precision_score(y_true, y_pred, average=None)):
        print("Precision Score for digit ", digit, ": {0:.3f}".format(score))


def get_error_rate(clf, test_data, test_label):

    print("Error Rate: ", "{0:.3f}".format(1 - clf.score(test_data, test_label)))


def get_recall_score(y_pred, y_true):
    for digit, score in enumerate(recall_score(y_true, y_pred, average=None)):
        print("Recall Score for digit ", digit, ": {0:.3f}".format(score))

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    clf_ada_boost = AdaBoost(train_data, train_labels)
    clf_mlp = MLP(train_data, train_labels)
    clf_svm = SVM(train_data, train_labels)

    # Metric Testing for AdaBoost
    y_pred = clf_ada_boost.predict(test_data)
    plot_roc_curve(clf_ada_boost, test_data, test_labels, "AdaBoost")
    print_accuracy_metric(clf_ada_boost, test_data, test_labels, train_data, train_labels)
    make_confusion_matrix(clf_ada_boost, test_data, test_labels, "AdaBoost")
    get_error_rate(clf_ada_boost, test_data, test_labels)
    get_precision_score(y_pred, test_labels)
    get_recall_score(y_pred, test_labels)

    # Metric Testing for SVM
    y_pred = clf_svm.predict(test_data)
    plot_roc_curve(clf_svm, test_data, test_labels, "SVM")
    print_accuracy_metric(clf_svm, test_data, test_labels, train_data,
                          train_labels)
    make_confusion_matrix(clf_svm, test_data, test_labels, "SVM")
    get_error_rate(clf_svm, test_data, test_labels)
    get_precision_score(y_pred, test_labels)
    get_recall_score(y_pred, test_labels)

    # Metric Testing for MLP
    y_pred = clf_mlp.predict(test_data)
    plot_roc_curve(clf_mlp, test_data, test_labels, "MLP")
    print_accuracy_metric(clf_mlp, test_data, test_labels, train_data,
                          train_labels)
    make_confusion_matrix(clf_mlp, test_data, test_labels, "MLP")
    get_error_rate(clf_mlp, test_data, test_labels)
    get_precision_score(y_pred, test_labels)
    get_recall_score(y_pred, test_labels)


if __name__ == '__main__':
    main()
