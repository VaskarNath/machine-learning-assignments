'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.model_selection import KFold
from scipy import stats as s
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, mean_squared_error, confusion_matrix, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import seaborn as sn


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm


        You should return the digit label provided by the algorithm
        '''
        distances = self.l2_distance(test_point)
        k_nearest_indices = {}
        for i in range(len(distances)):
            if len(k_nearest_indices) != k:
                k_nearest_indices[i] = distances[i]
            else:
                max_index = max(k_nearest_indices, key=k_nearest_indices.get)
                if distances[i] < distances[max_index]:
                    k_nearest_indices.pop(max_index)
                    k_nearest_indices[i] = distances[i]
        labels = []
        for key in k_nearest_indices:
            labels.append(self.train_labels[key])


        # Seeking for any ties of modal values between two different digit classes
        digit = int(s.mode(labels)[0])
        max_count = labels.count(digit)
        mode_array = []
        for i in range(0, 10):
            if labels.count(i) == max_count:
                mode_array.append(i)

        # Handling ties of multiple modal values by picking the nearest digit between the ties
        min_distance_in_mode_array = 1000 # assume max distance is less than 1000
        if len(mode_array) > 1:
            for key in k_nearest_indices:
                if self.train_labels[key] in mode_array:
                    if k_nearest_indices[key] < min_distance_in_mode_array:
                        min_distance_in_mode_array = k_nearest_indices[key]
                        digit = self.train_labels[key]
        else:
            digit = mode_array[0]
        return digit

    def predict(self, test_data, k):
        """
        Predict the labels for entire test_data
        """
        result = []
        for test_point in test_data:
            result.append(self.query_knn(test_point, k))
        return result

    def predict_proba(self, test_data, k):
        """
        Output matrix of probability of being each class for each test_point in test_data
        """
        result = []

        for test_point in test_data:

            # Initializing probability array so that every class has atleast a non-zero probability
            prob_array = np.full((10,), 0.01)

            # Getting k nearest labels
            distances = self.l2_distance(test_point)
            k_nearest_indices = {}
            for i in range(len(distances)):
                if len(k_nearest_indices) != k:
                    k_nearest_indices[i] = distances[i]
                else:
                    max_index = max(k_nearest_indices, key=k_nearest_indices.get)
                    if distances[i] < distances[max_index]:
                        k_nearest_indices.pop(max_index)
                        k_nearest_indices[i] = distances[i]
            labels = []
            for key in k_nearest_indices:
                labels.append(self.train_labels[key])

            # Getting prob_array for the individual test point
            for i in range(10):
                if labels.count(i) != 0:
                    prob_array[i] = labels.count(i) / k
            result.append(prob_array)

        return result


def cross_validation(train_data, train_labels, k_range=np.arange(1, 16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    accuracy_list = []
    for k in k_range:
        kf = KFold(10, shuffle=True, random_state=89)
        accuracy = 0
        num = 0
        for train_index, test_index in kf.split(range(len(train_data))):
            x_train, x_val, y_train, y_val = train_data[train_index], train_data[test_index], \
                                               train_labels[train_index], train_labels[test_index]
            knn = KNearestNeighbor(x_train, y_train)
            for j in range(len(x_val)):
                if knn.query_knn(x_val[j], k) == y_val[j]:
                    accuracy += 1
                num += 1
        print(accuracy, " ", num)
        accuracy_list.append(accuracy/num)
    for l in range(15):
        print("The average accuracy across folds for K = ", l + 1, ": ", float("{0:.3f}".format(accuracy_list[l])))
    return accuracy_list.index(max(accuracy_list)) + 1


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    accuracy = 0
    num = 0
    for j in range(len(eval_data)):
        if knn.query_knn(eval_data[j], k) == eval_labels[j]:
            accuracy += 1
        num += 1
    return accuracy/num


def plot_roc_curve(clf, k, test_data, test_labels, type):
    y_score = np.array(clf.predict_proba(test_data, k))

    y_test = label_binarize(test_labels, classes=range(10))

    # Computing ROC curve and ROC area for each class
    false_pos_rate = {}
    true_pos_rate = {}
    roc_auc = {}

    # Plotting ROC curve with legend describing the area under curve
    for i in range(10):
        false_pos_rate[i], true_pos_rate[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(false_pos_rate[i], true_pos_rate[i])
    for i in range(10):
        plt.plot(false_pos_rate[i], true_pos_rate[i], label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Rate of False Positives')
    plt.ylabel('Rate of True Positives')
    plt.title(
        'ROC Curve for each of the 10 Classes of Handwritten Digits')
    plt.legend(loc="lower right")
    plt.savefig(type + " ROC Curve.pdf")


def make_confusion_matrix(y_pred, y_test, type):
    matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(matrix, range(10), range(10))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, cmap="coolwarm", annot=True, fmt='.3g')
    plt.title("Confusion Matrix")
    plt.savefig(type + " Confusion Matrix.pdf")


def get_precision_score(y_pred, y_true):
    for digit, score in enumerate(precision_score(y_true, y_pred, average=None)):
        print("Precision Score for digit ", digit, ": {0:.3f}".format(score))


def get_error_rate(clf, k, test_data, test_label):
    print("Error Rate: ", "{0:.3f}".format(1 - classification_accuracy(clf, k, test_data, test_label)))


def get_recall_score(y_pred, y_true):
    for digit, score in enumerate(recall_score(y_true, y_pred, average=None)):
        print("Recall Score for digit ", digit, ": {0:.3f}".format(score))


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)
    k = cross_validation(train_data, train_labels)

    print("The value of optimal K: ", k)

    # print("Test accuracy: ", float("{0:.3f}".format(classification_accuracy(knn, 1, test_data, test_labels))))
    # print("Train accuracy: ", float("{0:.3f}".format(classification_accuracy(knn, 1, train_data, train_labels))))
    #
    # print("Test accuracy: ", float("{0:.3f}".format(classification_accuracy(knn, 15, test_data, test_labels))))
    # print("Train accuracy: ", float("{0:.3f}".format(classification_accuracy(knn, 15, train_data, train_labels))))

    print("Test accuracy: ", float("{0:.3f}".format(classification_accuracy(knn, k, test_data, test_labels))))
    print("Train accuracy: ", float("{0:.3f}".format(classification_accuracy(knn, k, train_data, train_labels))))

    y_pred = knn.predict(test_data, k)

    plot_roc_curve(knn, k, test_data, test_labels, "knn")
    make_confusion_matrix(y_pred, test_labels, "knn")
    get_error_rate(knn, k, test_data, test_labels)
    get_precision_score(y_pred, test_labels)
    get_recall_score(y_pred, test_labels)


if __name__ == '__main__':
    main()
