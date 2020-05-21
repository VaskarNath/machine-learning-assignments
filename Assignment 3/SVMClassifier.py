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
from sklearn.metrics import roc_curve, accuracy_score, precision_score, mean_squared_error, confusion_matrix, recall_score


def SVM(train_data, train_labels):
    '''
    Train a SVM by tuning the right hyper-parameter to maximize accuracy of the
    estimator
    '''
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                  {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4],
                   'C': [1, 10, 100, 1000]},
                  {'kernel': ['poly'], 'gamma': [1e-3, 1e-4],
                   'C': [1, 10, 100, 1000]}]
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


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    clf = SVM(train_data, train_labels)

    y_score = clf.decision_function(test_data)

    y_test = label_binarize(test_labels, classes=range(10))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),
                                              y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(10):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        'Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()





    print("Best Parameters: ", clf.best_params_)
    print("Test accuracy: ",
          classification_accuracy(clf, test_data, test_labels))
    print("Train accuracy: ",
          classification_accuracy(clf, train_data, train_labels))


if __name__ == '__main__':
    main()
