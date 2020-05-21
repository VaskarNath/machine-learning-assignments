from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import math
import matplotlib.pyplot as plt


def load_data():

    # collecting and formatting data

    df_t = pd.read_csv('data/clean_real.txt', names=["news_headline"])
    df_f = pd.read_csv('data/clean_fake.txt', names=["news_headline"])
    df_t["target"] = 1
    df_f["target"] = 0
    df = pd.concat([df_t, df_f], sort=False)
    df_x = df["news_headline"]
    df_y = df["target"]

    # transforming data into sparse matrix

    cv1 = CountVectorizer()
    df_x = cv1.fit_transform(df_x)

    # splitting data

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y,
                                                        test_size=0.3,
                                                        train_size=0.7,
                                                        shuffle=True,
                                                        random_state=23)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
                                                    test_size=0.5,
                                                    train_size=0.5,
                                                    shuffle=True,
                                                    random_state=23)
    return x_test, y_test, x_train, y_train, x_val, y_val, cv1


def select_tree_model():
    depth_values = [10, 25, 50, 85, 110]
    criterion = ['gini', 'entropy']
    best_criteria = ""
    best_depth = 0
    best_validation_accuracy = -1
    x_test, y_test, x_train, y_train, x_val, y_val, cv1 = load_data()

    # iterating through the different hyper-parameters to train the model and
    # collecting validation accuracy to decide which model to use

    for criteria in criterion:
        for depth in depth_values:
            model = tree.DecisionTreeClassifier(criterion=criteria,
                                                max_depth=depth,
                                                random_state=63)
            model.fit(x_train, y_train)
            actual = y_val.values.tolist()
            accuracy = model.score(x_val, actual)
            print(criteria + " with depth " + str(
                depth) + " has validation accuracy of " + str(accuracy))
            if accuracy > best_validation_accuracy:
                best_criteria = criteria
                best_depth = depth
                best_validation_accuracy = accuracy

    # recreating the best hyper-parameter model and test the model on the test
    # data

    best_model = tree.DecisionTreeClassifier(criterion=best_criteria,
                                             max_depth=best_depth,
                                             random_state=63)
    best_model.fit(x_train, y_train)
    test_accuracy = best_model.score(x_test, y_test.values.tolist())
    print("final model with criterion " + best_criteria + " and depth " +
          str(best_depth) + " has test accuracy of " + str(test_accuracy))

    # plotting the tree

    # tree.plot_tree(best_model, feature_names=cv1.get_feature_names(),
    #                max_depth=2, fontsize=8)
    # plt.show() Uncomment this lines if you want the tree to plot

    return 0


def entropy(x, total):
    if total != 0:
        if x/total == 0:
            return 0
        else:
            return -(x / total) * math.log2(x / total)
    return 0


def compute_information_gain(word):
    x_test, y_test, x_train, y_train, x_val, y_val, cv1 = load_data()
    word_in_real = 0.0
    word_not_in_real = 0.0
    word_in_fake = 0.0
    word_not_in_fake = 0.0

    # collecting information on the frequency of the word in fake new and real
    # news

    news_headlines, target_values = \
        cv1.inverse_transform(x_train), y_train.values.tolist()
    for i in range(len(news_headlines)):
        if word in news_headlines[i]:
            if target_values[i] == 1:
                word_in_real += 1.0
            else:
                word_in_fake += 1.0
        else:
            if target_values[i] == 1:
                word_not_in_real += 1.0
            else:
                word_not_in_fake += 1.0

    word_in_headline = word_in_real + word_in_fake
    word_not_in_headline = word_not_in_real + word_not_in_fake
    total = word_in_headline + word_not_in_headline
    num_fake = word_in_fake + word_not_in_fake
    num_real = word_in_real + word_not_in_real

    # calculating the separate components to calculate total information gain

    root_entropy = entropy(num_real, total) + \
        entropy(num_fake, total)
    no_word_entropy = entropy(word_not_in_fake, word_not_in_headline) + \
        entropy(word_not_in_real, word_not_in_headline)
    word_entropy = entropy(word_in_real, word_in_headline) + \
        entropy(word_in_fake, word_in_headline)
    leaf_entropy = word_entropy * (word_in_headline / total) + \
        no_word_entropy * (word_not_in_headline / total)
    information_gain = root_entropy - leaf_entropy

    print("The feature \"" + word + "\" has information gain of " +
          str(information_gain))

    return information_gain


def select_knn_model():
    x_test, y_test, x_train, y_train, x_val, y_val, cv1 = load_data()
    k_values = []
    train_error_arr = []
    val_error_arr = []
    best_k = 0
    best_accuracy = -1

    # Iterating through the different hyper-parameter and collecting validation
    # and test errors

    for k in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        train_target = y_train.values.tolist()
        train_predict = model.predict(x_train)
        train_error = sum((train_predict - train_target)**2)/train_predict.size
        val_target = y_val.values.tolist()
        val_accuracy = model.score(x_val, val_target)
        val_predict = model.predict(x_val)
        val_error = sum((val_predict - val_target)**2)/val_predict.size
        print("KNN with k = " + str(k) + " gives validation accuracy of "
              + str(val_accuracy))
        k_values.append(k)
        train_error_arr.append(train_error)
        val_error_arr.append(val_error)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_k = k

    # plotting graph

    fig, ax = plt.subplots()
    ax.set_xlim(21, 1)
    plt.plot(k_values, val_error_arr, label="Validation Error", marker=".",
             markersize=4, color='red', linewidth=2)
    plt.plot(k_values, train_error_arr, label="Training Error", marker=".",
             markersize=4, color='blue', linewidth=2)
    plt.title("Training Error vs. Validation Error as a Function of KNN")
    plt.xlabel("Number of Nearest Neighbour")
    plt.ylabel("Test Error")
    plt.xticks(range(21))
    plt.legend()
    plt.savefig('error_plot.pdf')

    # testing final model

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(x_train, y_train)
    actual = y_test.values.tolist()
    test_accuracy = final_model.score(x_test, actual)
    print("final model with k = " + str(best_k) + " has test accuracy of " +
          str(test_accuracy))
    return 0


if __name__ == "__main__":
    select_tree_model()
    select_knn_model()
    compute_information_gain("the")
    compute_information_gain("hillary")
    compute_information_gain("donald")
    compute_information_gain("trumps")
    compute_information_gain("energy")
    compute_information_gain("turnbull")
    compute_information_gain("le")
    compute_information_gain("era")
    compute_information_gain("black")
    compute_information_gain("2016")
    compute_information_gain("clinton")
    compute_information_gain("breitbart")
    compute_information_gain("election")
