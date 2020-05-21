from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)

        # TODO: Plot feature i against y
        plt.scatter(X[:, i], y, label="{0}".format(features[i]), s=3)
        plt.xlabel("{0}".format(features[i]))
        plt.ylabel("Price")
        plt.legend()

    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    # TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!

    X_T = np.transpose(X)
    XX_T = np.dot(X_T, X)
    X_Ty = np.dot(X_T, Y)
    w = np.linalg.solve(XX_T, X_Ty)
    return w


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    # Visualize the features
    visualize(X, y, features)

    # Normalize the X data
    max_col_val = np.max(X, axis=0)
    min_col_val = np.min(X, axis=0)
    max_min = max_col_val - min_col_val
    norm_X = np.zeros(shape=(len(X), len(X[0])))

    for i in range(len(X)):
        arr = X[i]
        norm_X[i] = (arr - min_col_val) / max_min

    # Inserting columns of ones for bias term
    norm_X = np.insert(norm_X, 0, 1, axis=1)

    # TODO: Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(norm_X, y,
                                                        test_size=0.3,
                                                        train_size=0.7,
                                                        shuffle=True,
                                                        random_state=54)
    # Fit regression model
    w = fit_regression(x_train, y_train)

    # Tabulate the weights with features
    print("bias: ", round(w[0], 3))
    for i in range(len(features)):
        print(features[i], ": ", round(w[i + 1], 3))

    # Compute fitted values, MSE, etc.
    y_pred = np.dot(x_test, w)
    mse = 0.5 * ((y_test - y_pred)**2).mean() # 1/2 is used in the square loss function in the slides
    r2 = r2_score(y_test, y_pred)
    mae = (abs(y_test - y_pred)).mean()
    print("MSE: ", round(mse, 3))
    print("MAE: ", round(mae, 3))
    print("R^2: ", round(r2, 3))


if __name__ == "__main__":
    main()

